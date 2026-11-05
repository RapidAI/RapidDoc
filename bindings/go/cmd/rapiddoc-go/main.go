// Command rapiddoc-go is a source-preview example, not a second parser.
package main

import (
	"flag"
	"fmt"
	rapiddoc "github.com/RapidAI/RapidDoc/bindings/go"
	"log"
	"os"
)

func main() {
	library := flag.String("library", "", "native SDK library (or RAPIDDOC_NATIVE_LIBRARY)")
	output := flag.String("output", "output/rapiddoc-go", "output directory")
	device := flag.String("device", "cpu", "cpu or cuda")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "Usage: rapiddoc-go [-library DLL/SO] [-output DIR] input.pdf")
		os.Exit(2)
	}
	engine, err := rapiddoc.New(rapiddoc.Config{LibraryPath: *library, Options: rapiddoc.Options{"device": *device}})
	if err != nil {
		log.Fatal(err)
	}
	if err = run(engine, flag.Arg(0), *output); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}
func run(engine *rapiddoc.Engine, input, output string) error {
	defer engine.Close()
	result, err := engine.Parse(input, nil, nil)
	if err != nil {
		return err
	}
	if err = result.WriteTo(output); err != nil {
		return err
	}
	fmt.Println("Saved:", output)
	return nil
}
