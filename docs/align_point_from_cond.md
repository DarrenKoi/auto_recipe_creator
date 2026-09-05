# cond.txt 의 흰 박스에서 align point 를 계산하는 방법

OM / SEM align 이미지에는 엔지니어가 recipe 등록 때 그린 **흰 네모 박스**가 있다.
그 박스 좌표가 이미지 옆 `cond.txt` 에 숫자로 적혀 있고, 코드는 그 숫자로
"이 template 이 라이브 화면에서 맞았을 때 **어디를 더블클릭해야 하는가**" 를
계산한다. 이 문서는 그 계산을 처음부터 끝까지 따라간다.

같은 내용을 그림과 인터랙티브 계산기로 보려면 `docs/align_point_from_cond.html`
을 브라우저로 연다.

> 용어 정리. 실제 cond.txt 키 이름은 `!Locator` 가 아니라 **`!Cursor_info`** 다
> (실데이터에는 끝 `o` 가 빠진 `!Cursor_inf` 로 적혀 있어 파서는 `cursor_inf`
> 접두로 찾는다). 코드에 `!Locator` 키를 읽는 곳은 없다.

---

## 0. 먼저 결론 한 줄

```
align point (rcp 이미지 위)  =  이미지 중심          ← 박스 중심이 아니다
offset                        =  이미지 중심 - 박스 중심
라이브 프레임의 클릭 지점      =  매칭 위치 + offset × 매칭 배율
```

박스는 "**이 근처가 유니크한 무늬다**" 라는 매칭 단서일 뿐이고, 장비가 기억하는
align point 는 등록 당시 화면의 **정중앙**이다. 그래서 박스가 중앙에서 조금
빗나가 그려져 있으면 그 차이(offset)를 따로 들고 다녀야 한다.

---

## 1. 재료: cond.txt 는 어디 있고 무엇이 적혀 있나

이미지마다 같은 폴더 안의 숨김 폴더에 짝이 있다.

```
align_img_from_rcp/
├─ IMAP0001.jpeg              # OM  align key
├─ .IMAP0001.jpeg/cond.txt    #   ↑ 의 조건
├─ IMAP0002.jpeg              # SEM align key
└─ .IMAP0002.jpeg/cond.txt
```

실제 샘플(OM, 512px):

```
# Observation condition
Scope                   OM
Magnification           104
Pixel                   512,512
!Cursor_info            -1,-1,-1,-1,-1,-1,1770,1770,3380,3330,-1,-1,...
```

`!Cursor_info` 는 콤마로 나뉜 긴 목록이고 우리가 읽는 자리는 정해져 있다
(0 부터 센다).

| 자리 | 뜻 | 없을 때 |
|---|---|---|
| `[4], [5]` | crosshair `(cx, cy)` | 둘 중 하나라도 `-1` |
| `[6], [7], [8], [9]` | 흰 박스 `(left, top, right, bottom)` | `[8]` 또는 `[9]` 가 `-1` |

위 샘플이면 박스 = `(1770, 1770, 3380, 3330)`, crosshair 없음.
파서는 `poc/workflow_3/align/cond_file.py` 의 `parse_cond` 이고 결과는
`CondInfo(scope, pixel, box_ltrb, crosshair_xy)` 다.

---

## 2. 좌표계: 숫자가 이미지보다 10배 크다

`Pixel 512,512` 인데 박스 값은 3380 까지 간다. `!Cursor_info` 의 좌표는
**이미지 픽셀의 10배 해상도**(cursor frame) 로 적혀 있다.

```
이미지 px = cursor 값 / 10          (OVERSAMPLE = 10)
(1770, 1770, 3380, 3330) → (177, 177, 338, 333)
```

이 나눗셈이 `clean_align_image.cursor_to_image` 한 곳에만 있다. 다른 곳에서
다시 나누지 않는다.

**보정 한 가지.** 로드한 이미지 크기가 `Pixel` 과 다르면(리사이즈 저장 등)
`/10` 만으로는 어긋난다. `cond_for_image(cond, image.shape)` 가 `로드 크기 /
Pixel` 비율로 박스와 crosshair 를 축별로 곱해 주고 `pixel` 을 로드 크기로
바꿔 둔다. 한 번 보정된 cond 는 다시 넣어도 값이 안 변한다(멱등). 그래서
여러 레이어가 겹쳐 불러도 이중 보정이 없다.

---

## 3. 계산: 박스 중심 → offset

`poc/workflow_3/align/cond_template.py`:

```python
def _cond_box_center(box_ltrb):
    l, t = cursor_to_image(box_ltrb[:2])      # /10
    r, b = cursor_to_image(box_ltrb[2:])
    return (l + r) / 2, (t + b) / 2

def cond_align_offset(box_ltrb, shape_hw):
    h, w = shape_hw[:2]
    bcx, bcy = _cond_box_center(box_ltrb)
    return round(w / 2 - bcx), round(h / 2 - bcy)   # 이미지 중심 - 박스 중심
```

샘플로 손계산:

```
박스(px)     l=177  t=177  r=338  b=333
박스 중심    ((177+338)/2, (177+333)/2) = (257.5, 255.0)
이미지 중심  (256, 256)
offset       (256 - 257.5, 256 - 255.0) = (-1.5, +1.0) → round → (-2, +1)
```

offset `(-2, +1)` 이란 "박스 중심에서 왼쪽 2px, 아래 1px 이 진짜 align point"
라는 뜻이다. 이 샘플은 박스가 거의 정중앙이라 offset 이 작지만, 중앙에서
빗나가 그린 박스라면 수십 px 이 나온다.

`round` 는 Python 의 반올림(짝수 쪽)이라 `-1.5 → -2` 다. 1px 차이라
매칭에는 의미가 없지만 손계산과 대조할 때 헷갈리지 않도록 적어 둔다.

**왜 crop 이 아니라 cond 숫자만으로 계산하나.** 예전 코드는 이미지 내용을
검출해 잡은 crop 의 중심에서 offset 을 뽑았다. 검출 crop 이 조금만 치우쳐도
offset 이 같이 오염됐고, 그게 miss 절반의 원인이었다. 지금은 offset 을
**cond.txt 의 박스 기하만으로** 정하고 crop 은 따로 만든다(decoupled). crop
을 어떻게 잡든 align point 는 안 움직인다.

---

## 4. 가드: 이 박스를 믿어도 되나

`check_cond_box(box_ltrb, shape)` 가 `('ok' | 'warn' | 'skip', 이유,
offset_norm)` 을 돌려준다. `offset_norm` 은 offset 길이를 **이미지 대각선**으로
나눈 값이라 crop 크기와 무관한 척도다.

| 판정 | 조건 | 뜻 |
|---|---|---|
| skip `box:degenerate` | 폭 또는 높이 ≤ 0 | 좌표가 뒤집혔거나 깨짐 |
| skip `box:out_of_bounds` | 박스가 이미지 밖 | Pixel 불일치 의심 |
| skip `box:too_small` | `min(변) - 4 < 16px` | inset 후 무늬가 안 남음 |
| skip `offset:too_far` | offset_norm > 0.38 | "박스 ≈ 중앙" 가정 붕괴, 엔지니어 검토 |
| warn `offset:far` | offset_norm > 0.25 | 쓰긴 쓰되 경고 |
| warn `box:small` | `min(변) - 4 < 24px` | 신호 약함 경고 |
| ok | 그 외 | |

skip 이면 박스를 버리고 **이미지 중앙 15% 면적** crop(`centered_area_crop`) 에
offset `(0, 0)` 으로 간다. cond.txt 자체가 없을 때도 같은 폴백이다.

---

## 5. template 만들기: 박스 테두리 지우고 안쪽만 오려낸다

`cond_template_crop(gray, cond)`:

1. **박스 테두리 선만 inpaint** 로 지운다(`clean_image`, 두께 1 / dilate 1 /
   반경 2). 흰 선이 라이브 화면에는 없으니 남겨 두면 매칭을 방해한다.
   rcp cond 에 crosshair 가 같이 있어도 그건 박스 안을 지나는 **실제
   무늬**라 지우지 않는다(`crosshair_xy=None` 으로 마스킹).
2. 박스 안쪽을 **사방 2px 씩 대칭 inset** 해서 오린다. 대칭이라 crop 중심이
   박스 중심과 같고, 3 절의 offset 과 정확히 맞아떨어진다. inset 하면 16px
   미만이 될 때만 inset 을 생략한다.
3. 오린 crop 을 `build_template(..., align_offset_xy=offset)` 에 넣는다.
   template 이 offset 을 **품고 다닌다**.

---

## 6. 실전: 라이브 프레임에서 클릭 지점까지

`correction.py` 의 primary 경로:

```python
ox, oy = template.align_offset_xy
align_x = result.best_xy[0] + round(ox * result.best_scale)
align_y = result.best_xy[1] + round(oy * result.best_scale)
cx, cy  = clamp_to_fov(align_x, align_y, fw, fh, margin)
controller.move_to_point(cx, cy)        # 더블클릭 = 그 점으로 recenter
```

- `best_xy` 는 매칭 엔진이 라이브 프레임에서 찾은 **template 중심**(= 박스
  중심이 화면에 나타난 자리).
- `best_scale` 은 매칭 때 쓴 배율. 라이브 프레임이 rcp 이미지와 크기가
  다르면 offset 도 같은 배율로 늘려야 한다. `offset (-2, 1)` 에 배율 1.5 면
  `(-3, 2)` 가 된다(반올림 후).
- `clamp_to_fov` 는 클릭 지점이 화면 가장자리 마진 안에 머물게 자른다.

offset 이 `(0, 0)` 이면(박스 없음 폴백, 또는 정확히 중앙에 그린 박스) 클릭
지점은 `best_xy` 그대로다.

좌표계는 이렇게 네 단계를 지난다.

```
cursor frame (×10)  ─/10─▶  rcp 이미지 px  ─offset 계산─▶  offset
                                                            │ ×best_scale
라이브 프레임 px  ◀─ best_xy + offset·scale ─┘
        │ rect/screenshot 비율 (DPI 125/150%)
스크린 px (실제 마우스 좌표, sem_monitor/controller 가 변환)
```

---

## 7. OM 과 SEM 은 무엇이 다른가

**계산은 완전히 같다.** `build_templates_from_assets` 가 `IMAP0001` 로
`templates["OM"]`, `IMAP0002` 로 `templates["SEM"]` 을 같은 `load_template`
으로 만든다. 다른 것은 `key_type` 하나이고, 이건 매칭 rerank 방식(OM = MIND,
SEM = ECC)을 가를 뿐 offset 계산에는 안 쓰인다.

실데이터에서 보이는 차이는 박스의 **크기**다.

| | OM (IMAP0001) | SEM (IMAP0002) |
|---|---|---|
| 배율 | 저배율 (예: 104) | 고배율 (K 단위) |
| 박스가 차지하는 비율 | 프레임의 ~35% 변 | 프레임의 80~100% |
| offset 의 의미 | 박스 밖 여백이 많아 offset 이 실제로 클릭 위치를 바꾼다 | 박스가 거의 프레임이라 offset 은 대개 몇 px |
| 매칭 난점 | 주기 무늬(비슷한 박스가 여러 개) | aperture problem(선 교차점만 유니크) |

어느 쪽이든 align point 는 **이미지 중심**이고, 박스는 그 중심을 찾기 위한
무늬 단서라는 점은 같다.

---

## 8. 직접 확인해 보기 (Mac, 실장비 없이)

```bash
uv run python - <<'EOF'
from poc.workflow_3.align.cond_file import parse_cond
from poc.workflow_3.align.cond_template import cond_align_offset, check_cond_box
cond = parse_cond(open("poc/workflow_2/docs/journals/260608/cond_sample.txt").read())
print(cond.box_ltrb)                                   # (1770, 1770, 3380, 3330)
print(cond_align_offset(cond.box_ltrb, (512, 512)))    # (-2, 1)
print(check_cond_box(cond.box_ltrb, (512, 512)))       # ('ok', 'ok', 0.003...)
print(cond_align_offset((600, 600, 2200, 2200), (512, 512)))  # (116, 116) 왼쪽 위로 치우친 박스
EOF
```

## 참고 코드

| 무엇 | 어디 |
|---|---|
| cond.txt 파싱, Pixel 불일치 보정 | `poc/workflow_3/align/cond_file.py` (`parse_cond`, `cond_for_image`) |
| `/10` 변환, 테두리 inpaint | `poc/workflow_3/align/clean_align_image.py` (`cursor_to_image`, `clean_image`) |
| 박스 중심, offset, 가드, template crop | `poc/workflow_3/align/cond_template.py` |
| rcp 이미지 → template (OM/SEM 둘 다) | `poc/workflow_3/align/templates.py` (`load_template`) |
| 라이브 프레임에서 offset 적용 + 클릭 | `poc/workflow_3/align/correction.py` (`correct_align_fail_auto`, primary 분기) |
| 반대 작업: crosshair 를 이미지 위에 그리기 | `docs/crosshair_overlay_from_cond.md` |
