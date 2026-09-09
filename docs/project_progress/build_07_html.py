"""07_ax_innovation_challenge.md -> 탭형 단일 HTML (이미지 base64 내장).

절(H2)마다 탭 하나. 절 안 소제목(h3/h4) 목차, 해시 기반 탭 복원, 키보드 탭 조작,
인쇄 시 전체 절 출력. 그림은 base64 로 내장해 HTML 한 파일만 전달하면 된다.

원본(07_ax_innovation_challenge.md)과 요약본(_short.md)을 함께 빌드한다(요약본이 없으면 건너뜀).

실행 (markdown 은 프로젝트 의존성이 아니라 1회성으로 얹는다):
    uv run --with markdown python docs/project_progress/build_07_html.py
"""
import base64, html, pathlib, re
import markdown

HERE = pathlib.Path(__file__).resolve().parent
SRCS = [HERE / "07_ax_innovation_challenge.md", HERE / "07_ax_innovation_challenge_short.md"]

CSS = """
:root{--ink:#1f2328;--muted:#59636e;--line:#d0d7de;--bg:#fff;--soft:#f6f8fa;--acc:#0b5cad;--tag:#eaf2fb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15.5px/1.75 "Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",system-ui,sans-serif;word-break:keep-all}
header{border-bottom:1px solid var(--line);background:var(--soft)}
header .in{max-width:1040px;margin:0 auto;padding:22px 24px 0}
header h1{margin:0 0 14px;font-size:1.45rem;line-height:1.35}
[role=tablist]{display:flex;flex-wrap:wrap;gap:4px}
[role=tab]{border:1px solid var(--line);border-bottom:none;background:#eef1f4;color:var(--muted);padding:9px 16px;border-radius:8px 8px 0 0;cursor:pointer;font:inherit;font-size:.95rem}
[role=tab][aria-selected=true]{background:var(--bg);color:var(--acc);font-weight:700;position:relative;top:1px}
main{max-width:1040px;margin:0 auto;padding:28px 24px 80px}
h2{font-size:1.35rem;border-bottom:2px solid var(--acc);padding-bottom:6px;margin:0 0 20px}
h3{font-size:1.12rem;margin:38px 0 12px;padding-left:10px;border-left:4px solid var(--acc)}
h4{font-size:1rem;margin:26px 0 8px}
p{margin:0 0 14px}
blockquote{margin:0 0 18px;padding:12px 18px;background:var(--soft);border-left:4px solid var(--line);color:var(--muted);font-size:.93rem}
blockquote p{margin:0 0 6px}
code{background:var(--tag);color:var(--acc);padding:1px 6px;border-radius:4px;font:.85em ui-monospace,Menlo,Consolas,monospace}
table{border-collapse:collapse;width:100%;margin:14px 0 22px;font-size:.92rem;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:8px 10px;vertical-align:top;text-align:left}
th{background:var(--soft);white-space:nowrap}
tr:nth-child(even) td{background:#fafbfc}
img{display:block;max-width:100%;margin:18px auto 26px;border:1px solid var(--line);border-radius:6px}
ul,ol{padding-left:26px}li{margin:4px 0}
hr{border:0;border-top:1px solid var(--line);margin:30px 0}
.toc{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:12px 18px;margin-bottom:26px;font-size:.92rem}
.toc span{font-weight:700;color:var(--muted)}.toc ol{margin:6px 0 0;padding-left:22px}.toc a{color:var(--acc);text-decoration:none}
.pager{display:flex;justify-content:space-between;margin-top:48px;padding-top:16px;border-top:1px solid var(--line)}
.pager button{background:none;border:1px solid var(--line);border-radius:6px;padding:8px 14px;cursor:pointer;font:inherit;color:var(--acc)}
.pager button:disabled{visibility:hidden}
.vh{position:absolute;left:-9999px}
.toc ul{list-style:disc;margin:2px 0;padding-left:20px;font-size:.88rem}
[role=tab]:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
@page{size:A4;margin:14mm 15mm}@media print{header [role=tablist],.pager,.toc{display:none}header .in{padding-top:8px}header h1{font-size:1.15rem;margin-bottom:6px}main{padding:10px 0 0}body{font-size:10.5pt;line-height:1.5}p{margin:0 0 8px}h2{font-size:1.2rem;margin-bottom:10px}h3{margin:18px 0 8px;font-size:1.02rem;page-break-after:avoid}h4{margin:12px 0 6px}table{display:table;overflow:visible;font-size:8.6pt;margin:8px 0 12px;line-height:1.4}th{white-space:normal}td,th{padding:4px 6px;word-break:break-word;min-width:5.5em}tr{page-break-inside:avoid}img{max-height:62mm;width:auto;margin:8px auto 12px;page-break-inside:avoid}img.fig-l{max-height:84mm}li{margin:2px 0}blockquote{padding:6px 12px;margin-bottom:10px;font-size:.9rem}section[hidden]{display:block!important}section{page-break-before:always}section[data-i="0"]{page-break-before:auto}}
"""

JS = r"""
const tabs=[...document.querySelectorAll('[role=tab]')],panels=[...document.querySelectorAll('[role=tabpanel]')],N=tabs.length;
const cur=()=>tabs.findIndex(t=>t.getAttribute('aria-selected')==='true');
function show(i,focus){
 tabs.forEach((t,k)=>{t.setAttribute('aria-selected',k===i);t.tabIndex=k===i?0:-1;});
 panels.forEach((p,k)=>p.hidden=k!==i);
 document.getElementById('prev').disabled=i===0;document.getElementById('next').disabled=i===N-1;
 if(focus)panels[i].querySelector('h2').focus({preventScroll:true});
}
function fromHash(){
 const h=decodeURIComponent(location.hash.slice(1));
 const m=h.match(/^s(\d+)$/);
 if(m){const i=+m[1];if(i<N){show(i,false);window.scrollTo(0,0);return;}}
 else if(h.startsWith('h-')){const el=document.getElementById(h);
  if(el){show(panels.indexOf(el.closest('[role=tabpanel]')),false);el.scrollIntoView({behavior:'smooth',block:'start'});return;}}
 show(0,false);if(h)history.replaceState(null,'','#s0');
}
const go=i=>{if(i>=0&&i<N){location.hash='s'+i;show(i,true);}};
tabs.forEach(t=>{t.onclick=()=>go(+t.dataset.i);
 t.onkeydown=e=>{const k={ArrowRight:cur()+1,ArrowLeft:cur()-1,Home:0,End:N-1}[e.key];
  if(k===undefined)return;e.preventDefault();const j=(k+N)%N;go(j);tabs[j].focus();};});
document.getElementById('prev').onclick=()=>go(cur()-1);
document.getElementById('next').onclick=()=>go(cur()+1);
addEventListener('hashchange',fromHash);fromHash();
"""


def build(src):
    out = src.with_suffix(".html")
    text = src.read_text(encoding="utf-8")
    # H2 로 절 분할. 첫 덩어리는 표지(제목 + 인용 메타).
    parts = re.split(r"^(?=## )", text, flags=re.M)
    front, sections = parts[0], parts[1:]
    title = re.match(r"# (.+)", front).group(1).strip()
    front_body = front.split("\n", 1)[1].replace("\n---\n", "\n")

    def render(md_text):
        h = markdown.markdown(md_text, extensions=["tables", "sane_lists", "attr_list"])
        # 이미지 내장
        def embed(m):
            p = HERE / m.group(1)
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'src="data:image/png;base64,{b64}"'
        return re.sub(r'src="(img/[^"]+\.png)"', embed, h)

    tabs = [("표지", "표지", front_body)]
    for s in sections:
        head, body = s.split("\n", 1)
        full = head[3:].strip()
        label = re.sub(r"\s*·\s*Operation Improvement 측면 서술", " · O/I", full)
        label = label.replace("정량적 + 정성적 성과", "성과")
        tabs.append((label, full, body.replace("\n---\n", "\n")))

    def h3_toc(md_text):
        heads = re.findall(r"^(###|####) (.+)$", md_text, flags=re.M)
        if sum(1 for lv, _ in heads if lv == "###") < 2:
            return ""
        out, open_sub = [], False
        for lv, t in heads:
            a = f'<a href="#{slug(t)}">{html.escape(t)}</a>'
            if lv == "###":
                if open_sub:
                    out.append("</ul></li>"); open_sub = False
                elif out:
                    out.append("</li>")
                out.append(f"<li>{a}")
            else:
                if not open_sub:
                    out.append("<ul>"); open_sub = True
                out.append(f"<li>{a}</li>")
        out.append("</ul></li>" if open_sub else "</li>")
        return f'<nav class="toc" aria-label="이 절의 목차"><span>이 절의 목차</span><ol>{"".join(out)}</ol></nav>'

    def slug(t):
        return "h-" + re.sub(r"[^0-9A-Za-z가-힣]+", "-", t).strip("-").lower()

    def add_ids(h):
        return re.sub(r"<(h[34])>(.+?)</h[34]>", lambda m: f'<{m.group(1)} id="{slug(html.unescape(re.sub("<.+?>", "", m.group(2))))}">{m.group(2)}</{m.group(1)}>', h)

    nav = "".join(
        f'<button role="tab" id="tab-{i}" aria-controls="panel-{i}" data-i="{i}" '
        f'aria-selected="{"true" if i == 0 else "false"}" tabindex="{0 if i == 0 else -1}">{html.escape(l)}</button>'
        for i, (l, _, _) in enumerate(tabs)
    )
    panels = "".join(
        f'<section role="tabpanel" id="panel-{i}" aria-labelledby="tab-{i}" data-i="{i}" {"" if i == 0 else "hidden"}>'
        f'{"<h2 tabindex=-1>" + html.escape(f) + "</h2>" if i else "<h2 tabindex=-1 class=vh>표지</h2>"}{h3_toc(b)}{add_ids(render(b))}</section>'
        for i, (l, f, b) in enumerate(tabs)
    )

    doc = f"""<!doctype html>
    <html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{html.escape(title)}</title><style>{CSS}</style></head>
    <body><header><div class="in"><h1>{html.escape(title)}</h1><div role="tablist">{nav}</div></div></header>
    <main>{panels}<div class="pager"><button id="prev">← 이전 절</button><button id="next">다음 절 →</button></div></main>
    <script>{JS}</script></body></html>"""
    out.write_text(doc, encoding="utf-8")
    print(f"[INFO] wrote {out} ({out.stat().st_size/1024:.0f} KB), tabs={[l for l,_,_ in tabs]}")


for _src in SRCS:
    if _src.exists():
        build(_src)
