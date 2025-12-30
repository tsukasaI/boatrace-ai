use actix_web::{web, HttpResponse, Responder};
use crate::models::{PredictRequest, PredictResponse, ExactaPrediction, ErrorResponse};
use crate::predictor::Predictor;

/// Predict race outcome
pub async fn predict_race(req: web::Json<PredictRequest>) -> impl Responder {
    // Validate request
    if req.entries.len() != 6 {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: "invalid_request".to_string(),
            message: "Exactly 6 racer entries required".to_string(),
        });
    }

    // Get predictions
    let predictor = Predictor::new();
    let position_probs = predictor.predict_positions(&req.entries);

    // Calculate exacta probabilities
    let mut exacta_predictions = predictor.calculate_exacta_probs(&position_probs);

    // Add odds and EV if provided
    if let Some(ref odds_data) = req.odds {
        for pred in &mut exacta_predictions {
            if let Some(odds) = odds_data.iter()
                .find(|o| o.first == pred.first && o.second == pred.second)
            {
                pred.odds = Some(odds.odds);
                pred.expected_value = Some(pred.probability * odds.odds);
                pred.is_value_bet = pred.expected_value.unwrap_or(0.0) > 1.0;
            }
        }
    }

    // Filter value bets
    let value_bets: Vec<ExactaPrediction> = exacta_predictions
        .iter()
        .filter(|p| p.is_value_bet)
        .cloned()
        .collect();

    let response = PredictResponse {
        date: req.date,
        stadium_code: req.stadium_code,
        race_no: req.race_no,
        position_probs,
        exacta_predictions,
        value_bets,
    };

    HttpResponse::Ok().json(response)
}

/// Predict exacta probabilities only
pub async fn predict_exacta(req: web::Json<PredictRequest>) -> impl Responder {
    if req.entries.len() != 6 {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: "invalid_request".to_string(),
            message: "Exactly 6 racer entries required".to_string(),
        });
    }

    let predictor = Predictor::new();
    let position_probs = predictor.predict_positions(&req.entries);
    let mut exacta_predictions = predictor.calculate_exacta_probs(&position_probs);

    // Add odds and EV if provided
    if let Some(ref odds_data) = req.odds {
        for pred in &mut exacta_predictions {
            if let Some(odds) = odds_data.iter()
                .find(|o| o.first == pred.first && o.second == pred.second)
            {
                pred.odds = Some(odds.odds);
                pred.expected_value = Some(pred.probability * odds.odds);
                pred.is_value_bet = pred.expected_value.unwrap_or(0.0) > 1.0;
            }
        }
    }

    // Return only top 10 by probability
    exacta_predictions.truncate(10);

    HttpResponse::Ok().json(exacta_predictions)
}
