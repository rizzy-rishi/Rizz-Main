import pathlib,sys
from playwright.sync_api import sync_playwright
R=pathlib.Path(__file__).parent.resolve()
H=int(sys.argv[1]);D=R/f"ovl/{H}";D.mkdir(parents=True,exist_ok=True)
N=int(20.2*30)
with sync_playwright() as p:
    b=p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg=b.new_page(viewport={"width":1080,"height":H})
    pg.goto((R/"overlay.html").as_uri()+f"?h={H}");pg.wait_for_function("window.READY===true");pg.evaluate("document.fonts.ready.then(()=>true)")
    for i in range(N):
        pg.evaluate(f"render({i/30})")
        pg.screenshot(path=str(D/f"{i:04d}.png"),omit_background=True)
    b.close()
print("ok",H,N)
