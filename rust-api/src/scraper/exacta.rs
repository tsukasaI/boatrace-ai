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
    let title_selector = Selector::parse("span.title7_mainLabel")
        .map_err(|e| ScraperError::ParseError(e.to_string()))?;

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

    // Find the odds table. The page has several `div.table1` tables (a race
    // navigation table, the 2連単 grid, the 2連複 grid); the odds grid is the
    // one whose header lists the six first-place boats (is-boatColorN), so we
    // select by that rather than taking the first table.
    let table_selector =
        Selector::parse("div.table1 table").map_err(|e| ScraperError::ParseError(e.to_string()))?;
    let thead_selector =
        Selector::parse("thead").map_err(|e| ScraperError::ParseError(e.to_string()))?;
    let th_selector = Selector::parse("th").map_err(|e| ScraperError::ParseError(e.to_string()))?;

    let header_boats = |table: &scraper::ElementRef| -> Vec<u8> {
        let mut boats: Vec<u8> = Vec::new();
        if let Some(thead) = table.select(&thead_selector).next() {
            for th in thead.select(&th_selector) {
                if let Some(boat) = get_boat_number_from_element(&th) {
                    if !boats.contains(&boat) {
                        boats.push(boat);
                    }
                }
            }
        }
        boats
    };

    let (table, first_boats) = document
        .select(&table_selector)
        .map(|t| {
            let boats = header_boats(&t);
            (t, boats)
        })
        .find(|(_, boats)| boats.len() == 6)
        .ok_or_else(|| ScraperError::ParseError("Could not find odds table".to_string()))?;

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
            if class.contains(&format!("is-boatColor{}", i))
                || class == format!("is-boatColor{}", i)
            {
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

    /// Regression: the odds page leads with a race-navigation `div.table1`
    /// (no boat-colored header) before the odds grid. The parser must skip the
    /// nav table and read the grid whose header lists the six first-place boats.
    #[test]
    fn test_parse_exacta_odds_skips_nav_table() {
        let html = r#"
            <span class="title7_mainLabel">2連単オッズ</span>
            <div class="table1"><table>
                <thead><tr>
                    <th class="is-thColor3">1R</th><th class="is-thColor3">2R</th>
                </tr></thead>
                <tbody><tr><td colspan="2">締切予定時刻</td><td>15:24</td></tr></tbody>
            </table></div>
            <div class="table1"><table>
                <thead><tr>
                    <th class="is-boatColor1">1</th><th class="is-boatColor1">A</th>
                    <th class="is-boatColor2">2</th><th class="is-boatColor2">B</th>
                    <th class="is-boatColor3">3</th><th class="is-boatColor3">C</th>
                    <th class="is-boatColor4">4</th><th class="is-boatColor4">D</th>
                    <th class="is-boatColor5">5</th><th class="is-boatColor5">E</th>
                    <th class="is-boatColor6">6</th><th class="is-boatColor6">F</th>
                </tr></thead>
                <tbody><tr>
                    <td class="is-boatColor2">2</td><td class="oddsPoint ">18.1</td>
                    <td class="is-boatColor1">1</td><td class="oddsPoint ">19.6</td>
                    <td class="is-boatColor1">1</td><td class="oddsPoint ">14.7</td>
                    <td class="is-boatColor1">1</td><td class="oddsPoint ">23.6</td>
                    <td class="is-boatColor1">1</td><td class="oddsPoint ">23.6</td>
                    <td class="is-boatColor1">1</td><td class="oddsPoint ">29.5</td>
                </tr></tbody>
            </table></div>
        "#;
        let odds = parse_exacta_odds(html).expect("should parse odds grid");
        assert_eq!(odds.get(&(1, 2)), Some(&18.1));
        assert_eq!(odds.get(&(2, 1)), Some(&19.6));
        assert_eq!(odds.get(&(3, 1)), Some(&14.7));
        assert_eq!(odds.get(&(6, 1)), Some(&29.5));
    }
}
