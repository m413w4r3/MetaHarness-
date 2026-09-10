#import "colors.typ": *

#let report-header = table(
  columns: (1fr, 5fr),
  stroke: 0.6pt + light-grey,
  align: (left, right),
  [
    #image(
      "chap.png",
    )
  ],
  [
    #align(right)[
      #stack(
        dir: ttb,
        spacing: 6pt,
        
      text(
        size: 13pt,
        weight: "bold",
        fill: dark,
      )[Bulletin n°XX],
      text(
        size: 13pt,
      )[Septembre 2026]
    )
    ]
  ],
)

#let report-footer = context table(
  columns: (1fr, auto, 1fr),
  inset: (x: 8pt, y: 4pt),
  stroke: 0.6pt + light-grey,

  align: (left, center, right),

  [
    #set text(
      size: 9pt,
    )
    Bulletin-CODE
  ],

  [],

  [
    #set text(
      size: 9pt,
    )
    Bulletin n°XX
  ],

  [
    #set text(
      size: 9pt,
    )
    Actualité des codes et infrastructures X
    ],

  [
    #set text(
      size: 8pt,
      weight: "bold",
      fill: dark,
    )

    #counter(page).display() #text("/") #counter(page).final().at(0)
  ],

  [#set text(
      size: 9pt,
    )
    Septembre 2026
    ],
)
