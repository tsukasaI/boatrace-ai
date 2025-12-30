use actix_web::{web, HttpResponse};
use std::sync::Arc;
use tracing::warn;

use crate::AppState;
use boatrace::error::{validate_entries_count, AppError};
use boatrace::models::{ExactaPrediction, PredictRequest, PredictResponse};

/// Predict race outcome
pub async fn predict_race(
    state: web::Data<Arc<AppState>>,
    req: web::Json<PredictRequest>,
) -> Result<HttpResponse, AppError> {
    // Validate request
    validate_entries_count(req.entries.len())?;

    // Get predictions using ONNX model, falling back on error
    let (position_probs, mut exacta_predictions) =
        if let Some(ref predictor_mutex) = state.predictor {
            let mut predictor = predictor_mutex.lock().unwrap();
            match predictor.predict_positions(&req.entries) {
                Ok(pos_probs) => {
                    let exacta_probs = predictor.calculate_exacta_probs(&pos_probs);
                    (pos_probs, exacta_probs)
                }
                Err(e) => {
                    warn!("ONNX prediction failed, using fallback: {}", e);
                    let pos_probs = state.fallback_predictor.predict_positions(&req.entries);
                    let exacta_probs = state.fallback_predictor.calculate_exacta_probs(&pos_probs);
                    (pos_probs, exacta_probs)
                }
            }
        } else {
            let pos_probs = state.fallback_predictor.predict_positions(&req.entries);
            let exacta_probs = state.fallback_predictor.calculate_exacta_probs(&pos_probs);
            (pos_probs, exacta_probs)
        };

    // Add odds and EV if provided
    if let Some(ref odds_data) = req.odds {
        for pred in &mut exacta_predictions {
            if let Some(odds) = odds_data
                .iter()
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

    Ok(HttpResponse::Ok().json(response))
}

/// Predict exacta probabilities only
pub async fn predict_exacta(
    state: web::Data<Arc<AppState>>,
    req: web::Json<PredictRequest>,
) -> Result<HttpResponse, AppError> {
    validate_entries_count(req.entries.len())?;

    // Get predictions using ONNX model, falling back on error
    let (position_probs, mut exacta_predictions) =
        if let Some(ref predictor_mutex) = state.predictor {
            let mut predictor = predictor_mutex.lock().unwrap();
            match predictor.predict_positions(&req.entries) {
                Ok(pos_probs) => {
                    let exacta_probs = predictor.calculate_exacta_probs(&pos_probs);
                    (pos_probs, exacta_probs)
                }
                Err(e) => {
                    warn!("ONNX prediction failed, using fallback: {}", e);
                    let pos_probs = state.fallback_predictor.predict_positions(&req.entries);
                    let exacta_probs = state.fallback_predictor.calculate_exacta_probs(&pos_probs);
                    (pos_probs, exacta_probs)
                }
            }
        } else {
            let pos_probs = state.fallback_predictor.predict_positions(&req.entries);
            let exacta_probs = state.fallback_predictor.calculate_exacta_probs(&pos_probs);
            (pos_probs, exacta_probs)
        };
    drop(position_probs); // unused in this endpoint

    // Add odds and EV if provided
    if let Some(ref odds_data) = req.odds {
        for pred in &mut exacta_predictions {
            if let Some(odds) = odds_data
                .iter()
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

    Ok(HttpResponse::Ok().json(exacta_predictions))
}
