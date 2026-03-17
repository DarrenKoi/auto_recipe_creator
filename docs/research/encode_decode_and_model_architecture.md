# Encode/Decode와 Encoder/Decoder 메모 (2026-03-17)

## 목적

이 문서는 아래 질문을 분리해서 정리한다.

1. `encode`와 `decode`는 무엇인가?
2. 왜 어떤 모델은 encoder만 있거나 decoder만 있는가?
3. 왜 어떤 서비스나 모델 인터페이스에서는 `encode` 또는 `decode` 중 하나만 보이는가?

헷갈리는 핵심 이유는 `encode/decode`가 **tokenizer 단계**와 **모델 아키텍처 단계**에서 서로 다른 뜻으로 쓰이기 때문이다.

## 1. 첫 번째 의미: tokenizer의 encode/decode

이 의미가 가장 실무적으로 자주 쓰인다.

- `encode`: 텍스트를 토큰 ID로 바꾸는 것
- `decode`: 토큰 ID를 다시 텍스트로 바꾸는 것

예를 들면 아래와 같다.

- 입력 문자열: `"open the settings window"`
- tokenizer `encode` 결과: `[4910, 279, 10412, 5454]`
- tokenizer `decode` 결과: `"open the settings window"`

즉, tokenizer는 사람이 읽는 문자열과 모델이 읽는 숫자 시퀀스 사이를 오가는 변환기다.

### 왜 필요한가

신경망 모델은 문자열 자체를 직접 처리하지 않는다. 모델이 실제로 받는 것은 숫자 ID 시퀀스다. 따라서 추론 시 기본 흐름은 아래와 같다.

1. 입력 텍스트를 `encode` 해서 token IDs로 만든다.
2. 모델이 이 token IDs를 embedding으로 바꾸고 계산한다.
3. 모델이 다음 token IDs를 생성한다.
4. 생성된 token IDs를 `decode` 해서 사람이 읽을 수 있는 텍스트로 바꾼다.

따라서 텍스트 생성 모델을 실행할 때 tokenizer `encode/decode`는 거의 항상 필요하다.

## 2. 두 번째 의미: 모델 아키텍처의 encoder/decoder

이건 tokenizer가 아니라 transformer 구조를 말한다.

- `encoder`: 입력 전체를 읽고 문맥 표현을 만든다.
- `decoder`: 이전 토큰과 필요한 문맥을 바탕으로 다음 토큰을 생성한다.

이 기준으로 모델은 보통 3가지로 나뉜다.

### 2.1 Encoder-only

예: `BERT` 계열

특징:

- 입력을 잘 이해하는 데 초점이 있다.
- 문장 분류, 토큰 분류, 검색, 임베딩 같은 작업에 적합하다.
- 보통 긴 텍스트를 한 토큰씩 생성하는 용도는 아니다.

즉, "읽고 이해"는 잘하지만 "계속 써 내려가기"는 기본 목적이 아니다.

### 2.2 Decoder-only

예: `GPT`, `Llama`, `Qwen`

특징:

- 다음 토큰 예측을 반복하는 방식으로 동작한다.
- 채팅, 생성, 코드 작성, 요약 같은 생성형 작업에 강하다.
- 오늘날 많은 LLM이 이 구조를 쓴다.

즉, "앞에 주어진 문맥을 이어서 생성"하는 데 최적화되어 있다.

### 2.3 Encoder-decoder

예: `T5`, `BART`

특징:

- encoder가 입력을 읽고
- decoder가 그것을 바탕으로 출력 시퀀스를 생성한다.

번역, 요약, 문장 변환처럼 `입력 시퀀스 -> 출력 시퀀스` 구조가 분명한 작업에 잘 맞는다.

## 3. 왜 어떤 모델은 encoder만 있거나 decoder만 있는가

이유는 간단하다. **모든 작업이 두 모듈을 다 필요로 하지는 않기 때문**이다.

### Encoder-only가 존재하는 이유

입력을 잘 이해해서 분류하거나 벡터화하는 것이 목적이면 굳이 토큰을 한 개씩 생성하는 decoder가 필요 없다.

예:

- 감정 분류
- 문서 검색용 임베딩
- 개체명 인식
- 랭킹 모델

이런 작업은 출력이 긴 자연어 문장이 아닐 수 있다. 그래서 encoder-only 구조가 더 단순하고 효율적이다.

### Decoder-only가 존재하는 이유

생성형 AI의 핵심 작업은 "다음 토큰 예측"이다. 이 목적에는 decoder-only 구조가 매우 잘 맞는다.

예:

- 채팅
- 코드 생성
- 긴 설명 생성
- 단계적 추론 형식의 출력

즉, 생성이 핵심이면 encoder를 따로 두지 않고 decoder-only로 가는 것이 구조적으로 단순하고 확장도 쉽다.

### Encoder-decoder가 존재하는 이유

입력과 출력이 분명히 분리된 작업에서는 encoder-decoder가 더 자연스럽다.

예:

- 번역
- 요약
- 문장 재작성
- 입력 문서를 읽고 다른 형식으로 변환

이 구조는 "입력을 읽는 단계"와 "출력을 생성하는 단계"를 분리해서 다루기 쉽다.

## 4. 자주 생기는 오해

### 오해 1. decoder-only 모델은 encode가 없다

아니다. **tokenizer encode는 여전히 있다.**

`decoder-only`라는 말은 아키텍처에 별도 `encoder` 블록이 없다는 뜻이다. 입력 텍스트를 token IDs로 바꾸는 tokenizer `encode`가 없다는 뜻이 아니다.

즉:

- tokenizer `encode/decode`: 보통 둘 다 있음
- 모델 architecture `encoder/decoder`: 둘 다 있을 수도 있고, 하나만 있을 수도 있음

### 오해 2. encoder-only 모델은 decode가 전혀 없다

상황에 따라 다르다.

- tokenizer 차원에서는 보통 `decode`가 있다.
- 하지만 모델 자체는 자연어 생성용 decoder가 없을 수 있다.

즉, token IDs를 문자열로 바꾸는 tokenizer `decode`와, 토큰을 생성하는 decoder 네트워크는 다른 개념이다.

## 5. 왜 어떤 서비스는 encode만 제공하거나 decode만 거의 안 보이는가

이건 아키텍처 문제라기보다 **제품 인터페이스 문제**인 경우도 많다.

### 5.1 Embedding 서비스

임베딩 모델은 보통 입력 텍스트를 벡터로 바꾸는 것이 목표다. 그래서 사용자 입장에서는 `encode`에 가까운 기능만 보인다.

- 텍스트 입력
- 내부적으로 tokenization
- 최종적으로 벡터 출력

이 경우 자연어 생성 결과를 다시 문자열로 `decode` 할 일이 거의 없다.

### 5.2 Generation API

채팅 API나 completion API는 내부에서 tokenizer `encode/decode`를 다 쓰지만, 사용자에게는 텍스트 입출력만 보일 수 있다. 즉, `decode`가 없는 것이 아니라 **API가 감춰 둔 것**이다.

### 5.3 Vision/Audio 모델

멀티모달 모델은 텍스트 말고 이미지/오디오 processor가 추가된다. 이런 경우에는 "tokenizer encode/decode"보다 "processor 전처리/후처리"가 더 전면에 보일 수 있다.

즉, 어떤 모델이 `encode` 또는 `decode` 중 하나가 없어 보이는 이유는 아래 둘 중 하나인 경우가 많다.

- 아키텍처적으로 encoder-only 또는 decoder-only라서
- 서비스 인터페이스가 내부 단계를 숨겨서

## 6. 실무적으로 어떻게 구분하면 되는가

헷갈리면 아래처럼 먼저 질문하면 된다.

1. 지금 말하는 `encode/decode`가 tokenizer 이야기인가?
2. 아니면 model architecture의 `encoder/decoder` 이야기인가?

빠른 기준은 이렇다.

- `text -> token IDs -> text`를 말하면 tokenizer `encode/decode`
- `BERT/GPT/T5` 같은 구조 차이를 말하면 model `encoder/decoder`

## 결론

`encode/decode`는 두 층위에서 쓰인다.

- tokenizer `encode/decode`: 문자열과 토큰 ID를 변환
- model `encoder/decoder`: 입력 이해와 출력 생성 구조를 담당

그리고 어떤 모델이 둘 중 하나만 가진 것처럼 보이는 이유는 보통 두 가지다.

- 실제 아키텍처가 encoder-only 또는 decoder-only이기 때문
- 서비스 API가 내부 tokenizer 단계를 사용자에게 노출하지 않기 때문

따라서 "이 모델은 encode가 없나?" 또는 "decode가 없나?"라는 질문은, 먼저 **tokenizer 기능을 묻는지, 아키텍처 모듈을 묻는지** 분리해서 보는 것이 가장 중요하다.
