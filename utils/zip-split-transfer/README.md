# zip-split-transfer

`.safetensors` 파일을 Windows에서 2 GB 이하 조각으로 분할해서 업로드하고, Linux에서 다시 합치는 예제.

## 핵심 정리

- 이 방식은 **전송용 byte split** 이다.
- `model.safetensors.part001`, `part002` 같은 임시 조각을 만든다.
- Linux에서 다시 합쳐서 원래의 `model.safetensors` 파일로 복원한 뒤 사용한다.
- 이 경우 `model.safetensors.index.json` 은 **수정하지 않는다**.

## 언제 index.json 을 수정하나

- **수정 안 함**: 업로드 편의를 위한 전송용 분할
- **수정 필요**: 실제 런타임 shard 파일(`model-00001-of-00004.safetensors`)로 재구성할 때

이 예제는 첫 번째 경우만 다룬다.

## 파일

- `split_safetensors_windows.py`: Windows에서 `.safetensors` 분할 + SHA-256 생성
- `join_safetensors_linux.py`: Linux에서 조각 합치기 + SHA-256 검증
- `zip_split_windows.py`: 여러 파일을 ZIP으로 묶어 분할해야 할 때 쓰는 대안
- `join_unzip_linux.py`: ZIP 대안의 Linux 쪽 복원 스크립트

## 사용법

1. `split_safetensors_windows.py` 상단의 경로를 실제 Windows 경로로 수정한다.
2. Windows에서 실행한다.

```bash
py split_safetensors_windows.py
```

3. 생성된 `model.safetensors.part001`, `part002`, ... 와 `model.safetensors.sha256` 를 Linux 서버로 업로드한다.
4. `config.json`, tokenizer 파일들, `model.safetensors.index.json` 같은 작은 파일은 그냥 일반 업로드한다.
5. `join_safetensors_linux.py` 상단의 경로를 실제 Linux 경로로 수정한다.
6. Linux에서 실행한다.

```bash
python join_safetensors_linux.py
```

7. 합쳐진 최종 `model.safetensors` 파일을 모델 로딩에 사용한다.

## 경로 예시

Windows:

```python
SOURCE_FILE = Path(r"C:\models\my-model\model.safetensors")
OUTPUT_DIR = Path(r"C:\transfer\my-model-upload")
```

Linux:

```python
PARTS_DIR = Path("/home/ubuntu/uploads/my-model-upload")
OUTPUT_FILE = Path("/home/ubuntu/models/model.safetensors")
```

## 주의할 점

- 분할된 `part001`, `part002` 파일은 직접 로딩하는 용도가 아니다.
- 반드시 Linux에서 모두 합쳐서 원본 `.safetensors` 로 복원해야 한다.
- 조각 파일 이름이나 순서가 바뀌면 안 된다.
- SHA-256 검증이 성공한 뒤에만 모델 로딩에 사용하는 게 안전하다.
