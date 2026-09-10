#import "../../UTILS/helpers.typ": article, noteAnalyste, styled-table, fn, ioc, ioc-list, vt

#let bold-words = (
  "Void Blizzard",
)

#let bold-regex = (
  "APT[0-9]{2}",
)

#let bold-pattern = (bold-words + bold-regex).join("|")

#show regex("\\b(" + bold-pattern + ")\\b"): word => {
  text(weight: "bold")[#word]
}


#article(
  category: "Article",
  number: "00",
  title: "[Groupe] Titre",

  events: (
    ("1er septembre 2026", [ref], "https://www.google.com"),
  ),
  overview: (
    ("Victimologie", []),
    ("Arsenal", []),
    ("Objectif principal", []),
  ),
  body: [
    
Synthèse objective

  #noteAnalyste[

    Notes
  ]

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

// #ioc-list(
//   title: [IOC originaux],
//   ips: [
//     127.0.0.1
//   ],
//   domains: [
//     example.com
//   ],
//   urls: [
//   ],
//   files: [
//     545ea2be9c242511bc145755cf4bda2485ade42966e096f8b4d3da2a230e8974
//     6b2c02bf82087a3ca5fb7ef8046554ff29ce85d52202bdcfae2b2653aede139a
//   ],
// )

// #ioc-list(
//   title: [IOC originaux],
//   ips: ("", ""),
//   domains: ("", ""),
//   urls: ("", ""),
//   files: ("", ""),
// )


])

