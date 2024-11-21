from fasthtml.common import *

app, rt = fast_app()

@rt('/')

def get():
    return Titled("Rio Hondo Streamgage Height 6 HR Forecast in Roswell, NM", Img(src="static/gage-rio-hondo-forecast.png"), P("www.thomasott.io"))

serve()