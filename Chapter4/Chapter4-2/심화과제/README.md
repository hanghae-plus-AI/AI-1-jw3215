# 심화과제

## 설명

- PDF 문서를 Vector DB에 저장하고, 책 내용에 기반한 문답을 하는 챗봇입니다.
- 다음 스크립트로 Vector DB에 문서를 저장합니다.
```bash
cd scripts
python ./save_to_db.py ../resources/sicp.pdf 
```

## 구동 영상

![demo](./resources/demo.gif)

## 개선 필요점
 
- [ ] 매 요청마다 문서 전체를 읽어 오는 듯함. 응답까지 시간이 오래 걸림
  - Chroma의 persist 적용 필요
