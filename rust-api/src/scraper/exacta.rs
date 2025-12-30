//! Exacta (2連単) odds HTML parser

use super::ScraperError;
use scraper::{Html, Selector};
use std::collections::HashMap;

/// Parse exacta odds from HTML
///
/// The table structure:
/// - Header row: boat 1 | name | boat 2 | name | ... (6 boats as 1st place)
/// - Body rows: 2nd boat | odds | 2nd boat | odds | ... (for each 1st place column)
///
/// Returns a map of (first, second) -> odds
pub fn parse_exacta_odds(html: &str) -> Result<HashMap<(u8, u8), f64>, ScraperError> {
    let document = Html::parse_document(html);
    let mut odds: HashMap<(u8, u8), f64> = HashMap::new();

    // Find the 2連単オッズ section
    let title_selector =
        Selector::parse("span.title7_mainLabel").map_err(|e| ScraperError::ParseError(e.to_string()))?;

    let mut found_title = false;
    for element in document.select(&title_selector) {
        if element.text().collect::<String>().contains("2連単オッズ") {
            found_title = true;
            break;
        }
    }

    if !found_title {
        return Err(ScraperError::ParseError(
            "Could not find 2連単オッズ title".to_string(),
        ));
    }

    // Find the table - look for table1 class containing the odds table
    let table_selector =
        Selector::parse("div.table1 table").map_err(|e| ScraperError::ParseError(e.to_string()))?;

    let table = document
        .select(&table_selector)
        .next()
        .ok_or_else(|| ScraperError::ParseError("Could not find odds table".to_string()))?;

    // Get 1st place boats from header
    let thead_selector =
        Selector::parse("thead").map_err(|e| ScraperError::ParseError(e.to_string()))?;
    let th_selector = Selector::parse("th").map_err(|e| ScraperError::ParseError(e.to_string()))?;

    let thead = table
        .select(&thead_selector)
        .next()
        .ok_or_else(|| ScraperError::ParseError("Could not find table header".to_string()))?;

    let mut first_boats: Vec<u8> = Vec::new();
    for th in thead.select(&th_selector) {
        if let Some(boat) = get_boat_number_from_element(&th) {
            if !first_boats.contains(&boat) {
                first_boats.push(boat);
            }
        }
    }

    if first_boats.len() != 6 {
        tracing::warn!("Expected 6 first boats, got {}", first_boats.len());
    }

    // Parse body rows
    let tbody_selector =
        Selector::parse("tbody").map_err(|e| ScraperError::ParseError(e.to_string()))?;
    let tr_selector = Selector::parse("tr").map_err(|e| ScraperError::ParseError(e.to_string()))?;
    let td_selector = Selector::parse("td").map_err(|e| ScraperError::ParseError(e.to_string()))?;

    let tbody = table
        .select(&tbody_selector)
        .next()
        .ok_or_else(|| ScraperError::ParseError("Could not find table body".to_string()))?;

    for row in tbody.select(&tr_selector) {
        let cells: Vec<_> = row.select(&td_selector).collect();

        // Cells come in pairs: (boat number, odds) for each 1st place column
        // 6 columns = 12 cells per row
        let mut cell_idx = 0;

        for first_boat in &first_boats {
            if cell_idx + 1 >= cells.len() {
                break;
            }

            let boat_cell = &cells[cell_idx];
            let odds_cell = &cells[cell_idx + 1];
            cell_idx += 2;

            // Get second boat number
            let second_boat = get_boat_number_from_element(boat_cell)
                .or_else(|| get_boat_number_from_text(boat_cell));

            if let Some(second) = second_boat {
                if let Some(odds_value) = parse_odds_value(odds_cell) {
                    if *first_boat != second {
                        odds.insert((*first_boat, second), odds_value);
                    }
                }
            }
        }
    }

    Ok(odds)
}

/// Extract boat number from element's CSS class
fn get_boat_number_from_element(element: &scraper::ElementRef) -> Option<u8> {
    let classes = element.value().classes().collect::<Vec<_>>();

    for class in classes {
        for i in 1..=6 {
            if class.contains(&format!("is-boatColor{}", i)) || class == format!("is-boatColor{}", i) {
                return Some(i);
            }
        }
    }

    None
}

/// Extract boat number from element's text content
fn get_boat_number_from_text(element: &scraper::ElementRef) -> Option<u8> {
    let text: String = element.text().collect::<String>().trim().to_string();

    if let Ok(num) = text.parse::<u8>() {
        if (1..=6).contains(&num) {
            return Some(num);
        }
    }

    None
}

/// Parse odds value from cell
fn parse_odds_value(element: &scraper::ElementRef) -> Option<f64> {
    // Look for oddsPoint class first
    let odds_point_selector = Selector::parse(".oddsPoint").ok()?;

    let text = if let Some(odds_span) = element.select(&odds_point_selector).next() {
        odds_span.text().collect::<String>()
    } else {
        element.text().collect::<String>()
    };

    // Clean and parse
    let cleaned = text
        .replace(",", "")
        .replace("欠場", "")
        .replace("取消", "")
        .trim()
        .to_string();

    if cleaned.is_empty() || cleaned == "-" {
        return None;
    }

    cleaned.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_td_fragment(html: &str) -> scraper::ElementRef<'static> {
        // Wrap in table to ensure proper parsing
        let full_html = format!("<html><body><table><tr>{}</tr></table></body></html>", html);
        // Leak to get 'static lifetime for testing
        let leaked: &'static str = Box::leak(full_html.into_boxed_str());
        let document: &'static Html = Box::leak(Box::new(Html::parse_document(leaked)));
        let td_selector = Selector::parse("td").unwrap();
        document.select(&td_selector).next().unwrap()
    }

    #[test]
    fn test_parse_odds_value_simple() {
        let td = parse_td_fragment(r#"<td><span class="oddsPoint">5.5</span></td>"#);
        assert_eq!(parse_odds_value(&td), Some(5.5));
    }

    #[test]
    fn test_parse_odds_value_with_comma() {
        let td = parse_td_fragment(r#"<td><span class="oddsPoint">1,234.5</span></td>"#);
        assert_eq!(parse_odds_value(&td), Some(1234.5));
    }

    #[test]
    fn test_parse_odds_value_cancelled() {
        let td = parse_td_fragment(r#"<td>欠場</td>"#);
        assert_eq!(parse_odds_value(&td), None);
    }

    #[test]
    fn test_get_boat_number_from_text() {
        let td = parse_td_fragment(r#"<td>3</td>"#);
        assert_eq!(get_boat_number_from_text(&td), Some(3));
    }

    #[test]
    fn test_get_boat_number_from_class() {
        let td = parse_td_fragment(r#"<td class="is-boatColor2">2</td>"#);
        assert_eq!(get_boat_number_from_element(&td), Some(2));
    }

    #[test]
    fn test_get_boat_number_invalid() {
        let td = parse_td_fragment(r#"<td>abc</td>"#);
        assert_eq!(get_boat_number_from_text(&td), None);
    }

    #[test]
    fn test_parse_exacta_odds_no_title() {
        let html = r#"<html><body><p>No odds here</p></body></html>"#;
        let result = parse_exacta_odds(html);
        assert!(result.is_err());
    }
}
