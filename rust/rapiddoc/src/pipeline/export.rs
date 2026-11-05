//! Finish the public document in Rust before crossing a language boundary.
use super::*;

impl Output {
    pub fn render_document(
        &mut self,
        make_mode: &str,
        image_path_prefix: &str,
        magika_runtime: Option<&Path>,
    ) -> Result<(), Error> {
        crate::legacy_page::with_host_page_budget(|| {
            self.render_document_inner(make_mode, image_path_prefix, magika_runtime)
        })
    }

    fn render_document_inner(
        &mut self,
        make_mode: &str,
        image_path_prefix: &str,
        magika_runtime: Option<&Path>,
    ) -> Result<(), Error> {
        let mut pages = self.metadata["pages"].take();
        if image_path_prefix != "images" {
            rewrite_image_prefix(&mut pages, image_path_prefix);
        }
        // Move the tree into one request and reuse it for both exporters. No
        // page clone, JSON text, or round trip through the host is necessary.
        let mut request = json!({
            "make_mode": make_mode,
            "image_path_prefix": image_path_prefix, "magika_runtime": magika_runtime,
        });
        request["pages"] = pages;
        let rendered = (|| {
            let markdown = crate::legacy_page::render(&request)?;
            request["make_mode"] = json!("content_list");
            let content = crate::legacy_page::render(&request)?;
            Ok::<_, Error>((markdown, content))
        })();
        self.metadata["pages"] = request["pages"].take();
        let (markdown, content) = rendered?;
        self.metadata["markdown"] = markdown;
        self.metadata["content_list_json"] = content;
        Ok(())
    }
}

fn rewrite_image_prefix(value: &mut Value, prefix: &str) {
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                if key == "html" && child.is_string() {
                    let html = child.as_str().unwrap();
                    *child =
                        Value::String(html.replace("src=\"images/", &format!("src=\"{prefix}/")));
                } else {
                    rewrite_image_prefix(child, prefix);
                }
            }
        }
        Value::Array(items) => {
            for child in items {
                rewrite_image_prefix(child, prefix);
            }
        }
        _ => {}
    }
}
