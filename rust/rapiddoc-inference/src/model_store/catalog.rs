//! Canonical, language-independent registry. Parser support is separate from
//! whether an artifact can be downloaded.
use super::{StoreError, StoreErrorKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CATALOG_JSON: &str = include_str!("../../models.json");

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub filename: String,
    pub sha256: String,
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub schema_version: u32,
    pub defaults: BTreeMap<String, String>,
    pub models: BTreeMap<String, Model>,
}

impl Catalog {
    pub fn from_json(json: &str) -> Result<Self, StoreError> {
        let catalog: Self = serde_json::from_str(json)
            .map_err(|e| StoreError::new(StoreErrorKind::InvalidInput, e.to_string()))?;
        if catalog.schema_version != 1 || catalog.models.is_empty() {
            return Err(StoreError::invalid("Unsupported/empty model catalog"));
        }
        for (id, model) in &catalog.models {
            if id.is_empty()
                || model.filename.is_empty()
                || matches!(model.filename.as_str(), "." | "..")
                || model.filename.contains(['/', '\\', ':'])
                || model.filename.chars().any(char::is_control)
                || model.sha256.len() != 64
                || !model.sha256.bytes().all(|c| c.is_ascii_hexdigit())
                || !(model.url.starts_with("https://") || model.url.starts_with("http://"))
            {
                return Err(StoreError::invalid(format!(
                    "Invalid model catalog entry: {id}"
                )));
            }
        }
        for (role, id) in &catalog.defaults {
            if !catalog.models.contains_key(id) {
                return Err(StoreError::invalid(format!(
                    "Unknown model {id} for role {role}"
                )));
            }
        }
        Ok(catalog)
    }

    pub fn bundled() -> Result<Self, StoreError> {
        Self::from_json(CATALOG_JSON)
    }

    pub fn model(&self, role_or_id: &str) -> Result<(&str, &Model), StoreError> {
        let id = self
            .defaults
            .get(role_or_id)
            .map(String::as_str)
            .unwrap_or(role_or_id);
        self.models
            .get_key_value(id)
            .map(|(id, model)| (id.as_str(), model))
            .ok_or_else(|| StoreError::invalid(format!("Unknown model ID or role: {role_or_id}")))
    }
}
