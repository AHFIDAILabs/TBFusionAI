"""
JSON file-based participant store.

FLAG FOR PRODUCTION REVIEW: This stores records in a local JSON file and audio
files on disk.  Replace with a relational or document database before deploying
to a multi-instance or cloud environment.
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


class ParticipantStore:
    def __init__(self):
        config = get_config()
        self._audio_dir = config.paths.participants_path / "audio"
        self._records_file = config.paths.participants_path / "participants.json"
        self._project_root = config.paths.project_root
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        audio_bytes: bytes,
        audio_filename: str,
        age: int,
        cough_duration: int,
        prior_tb_history: bool,
        hemoptysis: bool,
        weight_loss: bool,
        fever: bool,
        night_sweats: bool,
        prediction_result: Dict,
    ) -> Dict:
        try:
            participant_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat(timespec="seconds")
            cough_sound = self._save_audio(participant_id, audio_filename, audio_bytes)

            record = {
                "participantId": participant_id,
                "timestamp": timestamp,
                "coughSound": cough_sound,
                "age": age,
                "coughDuration": cough_duration,
                "priorTBHistory": prior_tb_history,
                "hemoptysis": hemoptysis,
                "weightLoss": weight_loss,
                "fever": fever,
                "nightSweats": night_sweats,
                "prediction": {
                    "result": prediction_result.get("prediction", ""),
                    "predictionClass": prediction_result.get("prediction_class", 0),
                    "probability": prediction_result.get("probability", 0.0),
                    "confidenceLevel": prediction_result.get("confidence_level", ""),
                    "recommendation": prediction_result.get("recommendation", ""),
                },
            }

            self._append_record(record)
            logger.info(f"Participant saved: {participant_id}")
            return record

        except Exception as e:
            logger.error(f"Failed to save participant: {e}")
            raise

    def _save_audio(
        self, participant_id: str, original_filename: str, audio_bytes: bytes
    ) -> str:
        safe_name = Path(original_filename).name
        dest = self._audio_dir / f"{participant_id}_{safe_name}"
        try:
            dest.write_bytes(audio_bytes)
        except OSError as e:
            logger.error(f"Failed to write audio {dest}: {e}")
            raise
        try:
            return str(dest.relative_to(self._project_root))
        except ValueError:
            return str(dest)

    def _append_record(self, record: Dict) -> None:
        with _lock:
            try:
                records = (
                    json.loads(self._records_file.read_text(encoding="utf-8"))
                    if self._records_file.exists()
                    else []
                )
                records.append(record)
                self._records_file.write_text(
                    json.dumps(records, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.error(f"Failed to write participants.json: {e}")
                raise
