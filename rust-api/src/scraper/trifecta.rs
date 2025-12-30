//! Trifecta (3連単) odds HTML parser

use super::ScraperError;
use scraper::{Html, Selector};
use std::collections::HashMap;

/// Parse trifecta odds from HTML
///
/// The table structure:
/// - Header row: 6 columns (1st place boats)
/// - For each column: 5 groups of 2nd place boats (rowspan=4)
/// - Each group: 4 rows for 3rd place options
/// - Cell triplets: (2nd place with rowspan) (3rd place) (odds)
///
/// Returns a map of (first, second, third) -> odds
pub fn parse_trifecta_odds(html: &str) -> Result<HashMap<(u8, u8, u8), f64>, ScraperError> {
    let document = Html::parse_document(html);
    let mut odds: HashMap<(u8, u8, u8), f64> = HashMap::new();

    // Find the 3連単オッズ section
    let title_selector =
        Selector::parse("span.title7_mainLabel").map_err(|e| ScraperError::ParseError(e.to_string()))?;

    let mut found_title = false;
    for element in document.select(&title_selector) {
        if element.text().collect::<String>().contains("3連単オッズ") {
            found_title = true;
            break;
        }
    }

    if !found_title {
        return Err(ScraperError::ParseError(
            "Could not find 3連単オッズ title".to_string(),
        ));
    }

    // Find the table
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
        return Err(ScraperError::ParseError(format!(
            "Expected 6 first boats, got {}",
            first_boats.len()
        )));
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

    // Track current 2nd place boat for each column (due to rowspan)
    let mut current_second: Vec<Option<u8>> = vec![None; 6];

    for row in tbody.select(&tr_selector) {
        let cells: Vec<_> = row.select(&td_selector).collect();

        // Process cells - structure varies due to rowspan
        let mut cell_idx = 0;

        for (col_idx, first_boat) in first_boats.iter().enumerate() {
            if cell_idx >= cells.len() {
                break;
            }

            let cell = &cells[cell_idx];

            // Check if this cell has rowspan (new 2nd place boat)
            let rowspan = cell.value().attr("rowspan");

            if rowspan.is_some() {
                // This is a 2nd place boat cell
                let second_boat = get_boat_number_from_element(cell)
                    .or_else(|| get_boat_number_from_text(cell));

                current_second[col_idx] = second_boat;
                cell_idx += 1;

                // Next cell is 3rd place boat (use TEXT, not class)
                if cell_idx >= cells.len() {
                    break;
                }
                let third_cell = &cells[cell_idx];
                let third_boat = get_boat_number_from_text(third_cell);
                cell_idx += 1;

                // Next cell is odds
                if cell_idx >= cells.len() {
                    break;
                }
                let odds_cell = &cells[cell_idx];
                let odds_value = parse_odds_value(odds_cell);
                cell_idx += 1;

                if let (Some(second), Some(third), Some(value)) =
                    (second_boat, third_boat, odds_value)
                {
                    odds.insert((*first_boat, second, third), value);
                }
            } else {
                // No rowspan - use current 2nd place boat
                let second_boat = current_second[col_idx];

                // This cell is 3rd place boat (use TEXT, not class)
                let third_boat = get_boat_number_from_text(cell);
                cell_idx += 1;

                // Next cell is odds
                if cell_idx >= cells.len() {
                    break;
                }
                let odds_cell = &cells[cell_idx];
                let odds_value = parse_odds_value(odds_cell);
                cell_idx += 1;

                if let (Some(second), Some(third), Some(value)) =
                    (second_boat, third_boat, odds_value)
                {
                    odds.insert((*first_boat, second, third), value);
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
    fn test_parse_trifecta_odds_no_title() {
        let html = r#"<html><body><p>No odds here</p></body></html>"#;
        let result = parse_trifecta_odds(html);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_boat_number_from_text() {
        let td = parse_td_fragment(r#"<td>4</td>"#);
        assert_eq!(get_boat_number_from_text(&td), Some(4));
    }

    #[test]
    fn test_get_boat_number_out_of_range() {
        let td = parse_td_fragment(r#"<td>7</td>"#);
        assert_eq!(get_boat_number_from_text(&td), None);
    }

    #[test]
    fn test_parse_odds_value_with_decimal() {
        let td = parse_td_fragment(r#"<td><span class="oddsPoint">123.4</span></td>"#);
        assert_eq!(parse_odds_value(&td), Some(123.4));
    }

    #[test]
    fn test_parse_odds_value_dash() {
        let td = parse_td_fragment(r#"<td>-</td>"#);
        assert_eq!(parse_odds_value(&td), None);
    }
}
