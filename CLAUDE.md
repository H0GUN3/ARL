# Claude Code 스펙 주도 개발 가이드

> **Spec-Driven Development (SDD)**: 공식적이고 상세한 명세를 AI 코드 생성의 기반으로 사용하는 개발 방법론

## 프로젝트 개요

**프로젝트**: API Rate Limiting 전략 비교 연구 (LSTM vs LinUCB)
**방법론**: Spec-Driven Development + Claude Code
**AI 도구**: Claude AI (via Claude Code)

---

## 핵심 문서 참조

### 📋 명세 및 설계
- @docs/SPEC.md - 프로젝트 명세서 및 요구사항
- @docs/API_DESIGN.md - API 및 모듈 인터페이스 설계
- @docs/FOLDER_STRUCTURE.md - 디렉토리 구조 정의

### 🔨 구현 가이드
- @docs/IMPLEMENTATION_GUIDE.md - 단계별 구현 절차
- @docs/TESTING_STRATEGY.md - 테스트 및 검증 전략
- @docs/AI_INTEGRATION.md - Claude AI 및 AI 도구 활용 방법

### 📖 참고 자료
- 검증.md - 방법론 검증 (기존 문서)
- AI문제.md - AI가 제시한 과제 및 해결책

---

## 스펙 주도 개발 워크플로우

```
1단계: 명세 작성
  ↓
2단계: API 설계 (명세 기반)
  ↓
3단계: 폴더 구조 정의
  ↓
4단계: AI를 이용한 코드 생성 (명세 → 코드)
  ↓
5단계: 테스트 및 검증
  ↓
6단계: 문서화 및 반복
```

---

## 빠른 시작

### 첫 번째 실행
1. @docs/SPEC.md 읽기 (프로젝트 요구사항 이해)
2. @docs/API_DESIGN.md 읽기 (인터페이스 이해)
3. @docs/FOLDER_STRUCTURE.md로 폴더 생성
4. @docs/IMPLEMENTATION_GUIDE.md 따라 구현

### Claude Code로 작업할 때
1. 특정 파일 또는 모듈에 대해 명확한 작업 지시
2. 반드시 해당하는 @docs/*.md 파일 참조
3. 명세에 맞지 않는 변경은 먼저 CLAUDE.md에서 승인받기

---

## 2025년 SDD 핵심 원칙

| 원칙 | 설명 |
|------|------|
| **Spec = Source of Truth** | 명세가 유일한 진실 공급원 |
| **AI-Ready Specs** | AI 코드 생성이 용이한 명세 작성 |
| **Executable Specs** | 명세 자체가 테스트 가능해야 함 |
| **Version Control** | 명세와 코드를 함께 관리 |
| **Documentation First** | 구현 전에 문서화 완료 |

---

## 파일 구조

```
limiting/
├── CLAUDE.md (이 파일)
├── 검증.md
├── AI문제.md
├── docs/
│   ├── SPEC.md
│   ├── API_DESIGN.md
│   ├── IMPLEMENTATION_GUIDE.md
│   ├── TESTING_STRATEGY.md
│   ├── AI_INTEGRATION.md
│   └── FOLDER_STRUCTURE.md
├── src/ (구현 코드)
├── data/ (데이터)
├── experiments/ (실험)
└── results/ (결과)
```

더 자세한 폴더 구조는 @docs/FOLDER_STRUCTURE.md 참조

---

## 주요 용어

- **SDD (Spec-Driven Development)**: 명세를 먼저 작성하고 그것을 기반으로 개발
- **OpenAPI**: REST API 명세 표준 형식
- **API Contract**: 클라이언트-서버 간 약속 (입력/출력 형식)
- **Executable Specs**: 자동 테스트 가능한 명세

---

## Claude Code 사용 시 주의사항

✅ **권장**:
- @docs 파일에서 명세 인용하며 작업 지시
- "명세에 따라 구현" 명확히 하기
- 변경 전에 CLAUDE.md에서 검토 요청

❌ **지양**:
- 명확한 명세 없이 즉흥적 개발
- 여러 버전의 요구사항 제시
- 명세와 충돌하는 변경

---

## 다음 단계

> **지금 해야 할 일**: @docs/SPEC.md 작성 → 명세 확정 → 구현 시작

