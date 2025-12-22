Font_ref_code

>> 화장품 ref 컷 만들때 사용한 폰트 데이터셋 생성 코드


cosmetic_data_making

>> 공병 + 로고 + 폰트 데이터셋 합성시키는 요청 보내는 코드
>> Batch API 기반이라서 구글 클라우드 버킷에 데이터가 존재함. (로컬에서 데이터셋을 다운로드할 필요 X)
>>
>> 데이터셋 업로드가 끝나면 정리해두겠습니다.


Directed_cut_making

>> GPT로 프롬포트 enhance해서 연출컷 생성
>> 작업중 (아직 업로드 X)

----------------------------------------------------------------------

pkg_brightness_classification

CV2 패키지를 이용해서, 가상의 화장품 용기의 밝기를 측정해서 LIGHT (밝음), DARK (어두움) 으로 분류하는 코드

<img width="552" height="196" alt="image" src="https://github.com/user-attachments/assets/bcb40df2-b7c7-42c1-b406-a53ec0d5b284" />

<img width="552" height="196" alt="image" src="https://github.com/user-attachments/assets/2caa341f-4a54-4544-8884-2e37a3e9ccf3" />

----------------------------------------------------------------------

Review_tool

폰트, 글자 불량 분류하는 GUI

### 📂 1. 전체 폴더 구조

> **경로:** `C:\Users\~\review_result_classified`

```text
review_result_classified/
│
├── SS/       <-- (Same Font, Same Char)
│
├── DS/       <-- (Diff Font, Same Char)
│
├── SD/       <-- (Same Font, Diff Char)
│
├── DD/       <-- (Diff Font, Diff Char)
│
└── Trash/    <-- (버리는 이미지)

```

---

### 📂 2. 각 폴더별 저장되는 내용

| 폴더명 | 단축키 | 의미 (분류 기준) | 상세 설명 |
| --- | --- | --- | --- |
| **`SS`** | `1` | **S**ame Font, **S**ame Char | 폰트도 같고, 글자도 같은 경우 |
| **`DS`** | `2` | **D**iff Font, **S**ame Char | 폰트는 다른데, 글자는 같은 경우 |
| **`SD`** | `3` | **S**ame Font, **D**iff Char | 폰트는 같은데, 글자가 다른 경우 |
| **`DD`** | `4` | **D**iff Font, **D**iff Char | 폰트도 다르고, 글자도 다른 경우 |
| **`Trash`** | Space | **Trash** | 품질이 낮거나 잘못된 이미지 |

분류 작업 시 파일이 해당 폴더로 들어감
