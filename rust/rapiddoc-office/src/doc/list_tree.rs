//! Semantic list tree builder. Ordinary blocks close a list; visuals retain it.
use crate::docx::body::{Block, Content, Kind};
#[derive(Default)]
pub struct Builder {
    identity: Option<u32>,
    paths: Vec<Vec<usize>>,
}
fn at<'a>(page: &'a mut [Block], path: &[usize]) -> &'a mut Block {
    let b = &mut page[path[0]];
    if path.len() == 1 {
        return b;
    }
    let Content::Blocks(children) = &mut b.content else {
        unreachable!()
    };
    at(children, &path[1..])
}
impl Builder {
    pub fn clear(&mut self) {
        self.identity = None;
        self.paths.clear();
    }
    pub fn push(&mut self, page: &mut Vec<Block>, mut block: Block, identity: Option<u32>) {
        if !matches!(block.kind, Kind::List | Kind::Index) {
            if identity.is_none() {
                self.paths.clear();
                self.identity = None;
            }
            page.push(block);
            return;
        }
        let level = block.ilevel.unwrap_or(0).clamp(0, 8) as usize;
        if self.identity != identity {
            self.paths.clear();
            self.identity = identity;
        }
        self.paths.truncate(level + 1);
        if let Some(path) = self.paths.get(level) {
            if at(page, path).attribute != block.attribute {
                self.paths.clear();
            }
        }
        while self.paths.len() < level + 1 {
            let depth = self.paths.len();
            let mut list = Block::new(block.kind, Content::Blocks(vec![]));
            list.attribute = block.attribute;
            list.ilevel = Some(depth as i64);
            list.start = block.start.clone();
            let path = if let Some(parent) = self.paths.last() {
                let parent_path = parent.clone();
                let parent = at(page, &parent_path);
                let Content::Blocks(children) = &mut parent.content else {
                    unreachable!()
                };
                let mut path = parent_path;
                path.push(children.len());
                children.push(list);
                path
            } else {
                let path = vec![page.len()];
                page.push(list);
                path
            };
            self.paths.push(path);
        }
        let list = at(page, self.paths.last().unwrap());
        if let (Content::Blocks(children), Content::Blocks(mut items)) = (
            &mut list.content,
            std::mem::replace(&mut block.content, Content::None),
        ) {
            children.append(&mut items);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn text(s: &str) -> Block {
        Block::new(Kind::Text, Content::Text(s.into()))
    }
    fn item(level: i64, kind: &'static str) -> Block {
        let mut b = Block::new(Kind::List, Content::Blocks(vec![text("item")]));
        b.attribute = Some(kind);
        b.ilevel = Some(level);
        b
    }
    #[test]
    fn ordinary_paragraph_is_not_absorbed() {
        let mut b = Builder::default();
        let mut page = vec![];
        b.push(&mut page, item(0, "ordered"), Some(1));
        b.push(&mut page, text("body"), None);
        b.push(&mut page, item(0, "ordered"), Some(1));
        assert_eq!(page.len(), 3);
        assert!(matches!(page[1].kind, Kind::Text));
    }
    #[test]
    fn nested_visual_preserves_stack_and_type_change_closes_it() {
        let mut b = Builder::default();
        let mut page = vec![];
        b.push(&mut page, item(0, "ordered"), Some(1));
        b.push(&mut page, Block::new(Kind::Image, Content::None), Some(1));
        b.push(&mut page, item(1, "unordered"), Some(1));
        assert_eq!(page.len(), 2);
        let Content::Blocks(children) = &page[0].content else {
            panic!()
        };
        assert!(matches!(children[1].kind, Kind::List));
        b.push(&mut page, item(0, "unordered"), Some(1));
        assert_eq!(page.len(), 3);
    }
}
