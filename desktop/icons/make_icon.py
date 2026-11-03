# Generates desktop/icons/icon.svg (32x32 pixel grid) and icon-small.svg (16x16 grid). Flat colours, hard edges, no gradients.
import sys
SKY, CREAM, INK, RED = "#6b99b3", "#f8efc9", "#2a1b14", "#c8442a"
def svg(n, rects, note):
    u = 1024 // n
    body = "\n".join(f'  <rect x="{x*u}" y="{y*u}" width="{w*u}" height="{h*u}" fill="{c}"/>' for x, y, w, h, c in rects)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024" width="1024" height="1024" shape-rendering="crispEdges">\n'
            f'  <!-- {note} -->\n  <defs><clipPath id="r"><rect width="1024" height="1024" rx="224"/></clipPath></defs>\n'
            f'  <g clip-path="url(#r)">\n  <rect width="1024" height="1024" fill="{SKY}"/>\n{body}\n  </g>\n</svg>\n')
def block(x, y, w, h, accent=None):                  # an expert: cream body, 1 brown shade column on the right
    r = [(x, y, w - 1, h, CREAM), (x + w - 1, y, 1, h, INK)]
    if accent is not None:
        r.append((x, accent, w, 1, RED))
    return r
big = []
big += [(16, 1, 1, 3, INK)]                                              # antenna
big += [(15, 4, 2, 28, CREAM), (17, 4, 1, 28, INK)]                      # mast
big += [(5, 13, 23, 1, CREAM), (5, 14, 23, 1, INK)]                      # upper platform
big += block(4, 5, 5, 8) + block(10, 8, 4, 5) + block(19, 8, 4, 5) + block(24, 5, 5, 8, accent=9)   # upper experts
big += [(7, 21, 19, 1, CREAM), (7, 22, 19, 1, INK)]                      # lower platform
big += block(8, 16, 5, 5) + block(20, 16, 5, 5)                          # lower experts
big += [(18, 27, 1, 1, INK)]                                             # dish strut
big += [(20, 24, 3, 1, CREAM), (19, 25, 4, 3, CREAM), (23, 25, 1, 3, INK), (20, 28, 2, 1, CREAM), (22, 28, 2, 1, INK)]   # a round dish, shaded on the right
big += [(15, 9, 3, 1, RED), (15, 18, 3, 1, RED), (15, 30, 3, 1, RED)]    # accent bands on the mast
small = []
small += [(8, 0, 1, 2, INK)]
small += [(7, 2, 2, 14, CREAM), (9, 2, 1, 14, INK)]
small += block(1, 3, 5, 6) + block(11, 3, 4, 5) + block(11, 10, 4, 5)
small += [(7, 11, 3, 1, RED)]
note = "moe-cache icon: a flat pixel-art mast (cream, shaded brown on the right) with stacked expert blocks and a dish; palette of the control page"
open(sys.argv[1] + "/icon.svg", "w").write(svg(32, big, note))
open(sys.argv[1] + "/icon-small.svg", "w").write(svg(16, small, note + "; simplified for 16-32 px"))
