// #import "../../UTILS/helpers.typ": article, noteAnalyste, styled-table, fn, ioc, ioc-list, vt



#noteAnalyste[
  Ceci est le contenu de ma note
]

#styled-table(
  (auto, auto, auto),
  cell-align: (left, left, left),
  
  [], [],  [],
  [], [], [],
  [], [], [],
  [], [], [],
)

#fn("google.com")
#ioc[`0909fcd018366edf`]
#vt[`ceci est une requête`]

#ioc-list(
  title: [IOC],
  ips: [
    127.0.0.1
  ],
  domains: [
    google.com
  ],
  files: [
    545ea2be9c242511bc145755cf4bda2485ade42966e096f8b4d3da2a230e8974
  ],
)

#figure(
  image("image.png", width: 90%),
  caption: [caption],
) <image>

#better-link(url)[#url]