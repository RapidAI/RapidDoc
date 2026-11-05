# Consistent Java / Go / Node output

All three result writers now interpret their argument as an output root and save
under `<root>/<source-stem>/<auto|txt|ocr|office>/`. Markdown, middle/content JSON
and drawing PDFs have the source-stem prefix. Asset names and their configured
image directory stay unchanged, preserving relative Markdown links.

Path inputs retain their source filename. Binary/reader inputs use `document`.
Office content is detected from the shared SDK's metadata and uses `office`;
PDF uses the requested parse method, including engine defaults and per-call
overrides. Structured Markdown uses `<stem>_markdown.json`.

Java `getOutputDirectory(root)`, Go `OutputDirectory(root)` and Node
`outputDirectory(root)` expose the final directory. Java engine automatic saving
uses the same writer without adding a second filename directory. Demo programs
pass their run root directly and no longer prepend numbered filename folders.

This intentionally changes the preview writers' previous flat output layout.
Old files are not removed or moved. Parsing, model selection and native result
contents are unchanged. No Rust build was needed.

Validation: Java two affected native writer tests; Go reference/owned-output and
writer tests; Node seven-format/owned-assets and input/writer tests; TypeScript
declarations. Java jar installed into both D:/mavenlib and the user's .m2;
Node package rebuilt and installed into the independent WebProjects demo.
Go DOCX and Node real-PDF first-page demo runs saved the new layout successfully.
No full regression or CUDA run performed.
