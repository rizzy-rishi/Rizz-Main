import pathlib
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont
R=pathlib.Path(__file__).parent.resolve(); O=R/"out-geo"; O.mkdir(exist_ok=True)
with sync_playwright() as p:
    b=p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg=b.new_page(viewport={"width":1300,"height":1300})
    pg.goto((R/"geo.html").as_uri());pg.wait_for_function("window.READY===true");pg.evaluate("document.fonts.ready.then(()=>true)");pg.wait_for_timeout(400)
    names=pg.evaluate("window.NAMES")
    for i in range(len(names)):
        pg.locator(f"#g{i+1}").screenshot(path=str(O/f"geo-{i+1:02d}.png"))
    b.close()
f=ImageFont.truetype(str(R/"fonts/Montserrat_wght_800_.ttf"),30)
T=560;pad=30;lab=60;n=len(names);cols=4;rows=(n+cols-1)//cols
board=Image.new("RGB",(cols*T+(cols+1)*pad,rows*(T+lab)+(rows+1)*pad),"#1d2326");d=ImageDraw.Draw(board)
for i in range(n):
    im=Image.open(O/f"geo-{i+1:02d}.png").convert("RGB").resize((T,T),Image.LANCZOS)
    x=pad+(i%cols)*(T+pad);y=pad+(i//cols)*(T+lab+pad)
    board.paste(im,(x,y));d.text((x,y+T+14),f"{chr(65+i)}  {names[i].upper()}",font=f,fill="#E8B45A")
board.save(O/"GRG-geometric-options-board.png");print(names)
