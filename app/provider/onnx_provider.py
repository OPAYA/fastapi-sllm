import logging
from typing import Dict, Any, Optional
import os
import time
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from app.config import get_settings
from app.provider.model_provider import ModelProvider

logger = logging.getLogger(__name__)

class OnnxModelProvider(ModelProvider):
    """
    중앙 ONNX 모델 제공자 클래스
    모든 버전의 SLLM 서비스가 공유할 수 있는 ONNX 모델 인스턴스를 관리합니다.
    싱글톤 패턴을 구현하되 global 변수를 사용하지 않습니다.
    """
    _instance: Optional['OnnxModelProvider'] = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("OnnxModelProvider 인스턴스 생성")
            cls._instance = super(OnnxModelProvider, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, "_initialized", False):
            logger.info("OnnxModelProvider 초기화")
            settings = get_settings()
            self.model_path = settings.ONNX_MODEL_PATH
            self.default_model = settings.DEFAULT_ONNX_MODEL
            self.session = None
            self.model_loaded = False
            self.model_load_time = 0
            self.tokenizer = None

            # 모델 로드 시도
            self.load_model()
            self._initialized = True

    def load_model(self) -> bool:
        """ONNX 모델을 로드합니다."""
        if not os.path.exists(self.model_path):
            logger.warning(f"ONNX 모델 파일을 찾을 수 없습니다: {self.model_path}")
            logger.warning("모델이 로드되지 않았습니다.")
            return False

        try:
            logger.info(f"ONNX 모델 로딩 중: {self.model_path}")
            start_time = time.time()

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

            # ONNX Runtime 세션 옵션 설정
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 4

            # ONNX 모델 로드
            self.session = ort.InferenceSession(
                self.model_path,
                session_options,
                providers=['CPUExecutionProvider']
            )

            self.model_load_time = time.time() - start_time
            self.model_loaded = True
            logger.info(f"ONNX 모델 로딩 완료! 소요 시간: {self.model_load_time:.2f}초")

            # 모델 정보 출력
            input_names = [input.name for input in self.session.get_inputs()]
            output_names = [output.name for output in self.session.get_outputs()]
            logger.info(f"모델 입력: {input_names}")
            logger.info(f"모델 출력: {output_names}")

            return True
        except Exception as e:
            logger.error(f"ONNX 모델 로딩 실패: {str(e)}")
            logger.warning("모델이 로드되지 않았습니다.")
            return False

    def is_loaded(self) -> bool:
        """모델이 로드되었는지 확인합니다."""
        return self.model_loaded and self.session is not None

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다."""
        return {
            "model_loaded": self.model_loaded,
            "model_load_time": self.model_load_time,
            "model_path": self.model_path,
            "default_model": self.default_model
        }

    def run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ONNX 모델을 사용하여 추론을 실행합니다.

        Args:
            input_data (Dict[str, Any]): 모델 입력 데이터

        Returns:
            Dict[str, Any]: 모델 출력 데이터
        """
        logger.info(f"onnx run_inference 호출")
        if not self.is_loaded():
            logger.error("모델이 로드되지 않았습니다.")
            return None

        try:
            # 입력 데이터 준비
            prompt = input_data.get("prompt", "")
            temperature = input_data.get("temperature", 0.7)
            max_tokens = input_data.get("max_tokens", 1000)

            # 토크나이저로 입력 처리
            inputs = self.tokenizer(prompt, return_tensors="np", padding=False)
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

            # 초기 생성 ID 설정
            generated_ids = input_ids.copy()

            # 종료 토큰 ID 가져오기
            eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

            # 출력 이름 가져오기
            output_names = [o.name for o in self.session.get_outputs()]

            logger.info(f"🧠 생성 시작: 최대 {max_tokens} 토큰")

            # 토큰 생성 루프
            for step in range(max_tokens):
                seq_len = generated_ids.shape[1]

                # position_ids 직접 생성
                position_ids = np.arange(seq_len)[None, :].astype(np.int64)

                # ONNX 입력 구성
                onnx_inputs = {
                    "input_ids": generated_ids.astype(np.int64),
                    "attention_mask": np.ones_like(generated_ids, dtype=np.int64),
                    "position_ids": position_ids,
                }

                # 추론
                outputs = self.session.run(output_names, onnx_inputs)
                logits = outputs[0]  # shape: (1, seq_len, vocab_size)

                # 마지막 토큰에서 최고 확률 토큰 추출
                next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])

                # 종료 조건
                if next_token_id == eos_token_id:
                    logger.info("🛑 종료 토큰 생성됨!")
                    break

                # 다음 토큰 추가
                next_token_arr = np.array([[next_token_id]])
                generated_ids = np.concatenate([generated_ids, next_token_arr], axis=-1)

                logger.info(f"step: {step}")

            # 결과 디코딩
            output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logger.info(f"📘 생성된 문장: {output_text}")

            return {
                "text": output_text,
                "usage": {
                    "prompt_tokens": len(input_ids[0]),
                    "completion_tokens": len(generated_ids[0]) - len(input_ids[0]),
                    "total_tokens": len(generated_ids[0])
                }
            }
        except Exception as e:
            logger.error(f"추론 실행 중 오류 발생: {str(e)}")
            return None

    def close(self) -> None:
        """모델 리소스를 해제합니다."""
        if self.session is not None:
            # ONNX 세션은 명시적인 close 메서드가 없으므로 참조를 제거
            self.session = None
            self.model_loaded = False

def get_onnx_session() -> ort.InferenceSession:
    """
    ONNX 세션을 가져옵니다.
    """
    provider = OnnxModelProvider()
    return provider.session if provider.is_loaded() else None

def get_onnx_model_info() -> dict:
    """
    ONNX 모델 정보를 반환합니다.
    """
    provider = OnnxModelProvider()
    return provider.get_model_info()
