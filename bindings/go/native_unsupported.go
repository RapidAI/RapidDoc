//go:build (!windows && !linux) || !cgo

package rapiddoc

import "errors"

func terminalOutput() bool { return false }

func openLibrary(string) (*nativeAPI, error) {
	return nil, errors.New("supported native hosts: Windows/Linux amd64 with CGO_ENABLED=1 and a C compiler")
}
