import pathlib
from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw, ImageFont
R=pathlib.Path(__file__).parent.resolve()
NAMES=["Classic Capital","Horizon Split","Tower Stack","Gold Foil Luxury","Blueprint","Constructed Geometric","Skyline Windows","Vibrant Gradient","Horizontal Lockup","Monoline Wide","Heritage Seal","Roof Line"]
with sync_playwright() as p:
    b=p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg=b.new_page(viewport={"width":1300,"height":1300},device_scale_factor=1)
    pg.goto((R/"index.html").as_uri());pg.wait_for_function("window.READY===true");pg.evaluate("document.fonts.ready.then(()=>true)");pg.wait_for_timeout(500)
    for i in range(1,13):
        pg.locator(f"#o{i}").screenshot(path=str(R/f"out/option-{i:02d}.png"))
    b.close()
f=ImageFont.truetype(str(R/"fonts/Montserrat_wght_800_.ttf"),30)
T=560;pad=30;lab=60
board=Image.new("RGB",(4*T+5*pad,3*(T+lab)+4*pad),"#1d2326");d=ImageDraw.Draw(board)
for i in range(12):
    im=Image.open(R/f"out/option-{i+1:02d}.png").convert("RGB").resize((T,T),Image.LANCZOS)
    x=pad+(i%4)*(T+pad);y=pad+(i//4)*(T+lab+pad)
    board.paste(im,(x,y));d.text((x,y+T+14),f"{i+1:02d}  {NAMES[i].upper()}",font=f,fill="#E8B45A")
board.save(R/"out/GRG-logo-options-board.png")
print("done")
