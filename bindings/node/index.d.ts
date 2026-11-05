/// <reference types="node" />
export type Input = string | URL | Uint8Array;
/** Shared Rust options; end_page is exclusive. Unknown native fields are rejected. */
export interface ParseOptions {
  start_page?: number;
  end_page?: number;
  parse_method?: string;
  formula_enable?: boolean;
  table_enable?: boolean;
  lang?: string;
  image_dir_name?: string;
  doc_orientation_classify?: boolean;
  draw_layout?: boolean;
  draw_span?: boolean;
  draw_line_sort?: boolean;
  [key: string]: unknown;
}
export interface ProgressEvent {
  stage: string;
  page_index: number | null;
  completed: number | null;
  total: number | null;
  elapsed_seconds: number | null;
  detail: string | null;
}
export type Progress = (event: ProgressEvent) => void | Promise<void>;
export interface Config {
  libraryPath?: string;
  runtimeDirectory?: string;
  config?: Record<string, unknown>;
  resources?: {
    bundle_dirs?: string[];
    model_directory?: string;
    allow_download?: boolean;
    ort_library?: string;
    pdfium_library?: string;
    geos_library?: string;
    operator_library?: string;
  };
  progressLogging?: boolean;
  progressMode?: "auto" | "bar" | "plain";
}
export class RapidDocError extends Error {
  constructor(code: string, message: string);
  code: string;
}
export class DocumentResult {
  private constructor();
  rawJSON: string;
  markdown: string | unknown[];
  middleJSON: Record<string, unknown>;
  contentListJSON: unknown[];
  metadata: Record<string, unknown>;
  diagnostics: string[];
  sourcePageCount: number;
  readonly sourceName: string;
  readonly parseMethod: string;
  images: Record<string, Buffer>;
  drawings: Record<string, Buffer>;
  writeTo(directory: string): Promise<void>;
  outputDirectory(directory: string): string;
}
/** String input accepts local paths and HTTP/HTTPS URLs; downloads run in Rust. */
export class RapidDoc {
  constructor(options?: Config);
  parse(
    input: Input,
    options?: ParseOptions,
    progress?: Progress | null,
    signal?: AbortSignal,
  ): Promise<DocumentResult>;
  parse(
    input: Input[],
    options?: ParseOptions,
    progress?: Progress | null,
    signal?: AbortSignal,
  ): Promise<DocumentResult[]>;
  parseAll(
    inputs: Input[],
    options?: ParseOptions,
    progress?: Progress | null,
    signal?: AbortSignal,
  ): Promise<DocumentResult[]>;
  parseReader(
    reader: AsyncIterable<Uint8Array>,
    options?: ParseOptions,
    progress?: Progress | null,
    signal?: AbortSignal,
  ): Promise<DocumentResult>;
  warmup(
    options?: ParseOptions,
    progress?: Progress | null,
    signal?: AbortSignal,
  ): Promise<string[]>;
  loadedModels(): Promise<string[]>;
  close(): Promise<void>;
}
