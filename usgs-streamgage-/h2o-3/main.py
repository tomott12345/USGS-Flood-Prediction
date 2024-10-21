from fasthtml.common import *

app, rt = fast_app()

@rt('/')

def get():
    return Titled("Flood Forecast for Pompton River @ Route 23, Pompton Plains NJ", Img(src="static/forecast.png"), P("www.thomasott.io"))

serve()