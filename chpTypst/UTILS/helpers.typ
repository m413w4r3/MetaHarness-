#import "colors.typ": *

#import "@preview/notionly:0.1.0": *

#import "@preview/oasis-align:0.4.1": *
#set grid(gutter: 1em)


// Petit label de section
#let tag(content) = box(
  fill: purple-light,
  inset: (x: 8pt, y: 4pt),
  radius: 3pt,
)[
  #set text(
    size: 8pt,
    weight: "bold",
    fill: black,
    tracking: 0.5pt,
  )

  #upper(content)
]

#let section-title(content) = [
  #v(4pt)

  #text(
    size: 14pt,
    weight: "extrabold",
    fill: purple,
  )[
    #content
  ]

]

#let better-link(target, body) = {
  if type(target) == str and not str.starts-with(target, "http") {
    body
  } else {
    text(
      fill: rgb("#0563C1"),
    )[
      #underline(
        link(target)[#body]
      )
    ]
  }
}

// Chronologie
#let timeline(events) = [
  #for event in events {
    grid(
      columns: (0.25cm, auto, 1fr),
      gutter: 2pt,
      pad(left: 2em)[
        #text(
          size: 11pt,
          fill: dark,
          weight: "bold",
        )[•]
      ],
      pad(left: 2em)[
        #text(
          size: 11pt,
          weight: "extrabold",
          fill: purple,
        )[
          #event.at(0)
        ]
        : #event.at(1)
        #if event.len() > 2 and event.at(2) != none and event.at(2) != "" {
            footnote[#better-link(event.at(2))[#event.at(2)]]
          }
      ]
    )

  }
]

#let separator() = line(
  length: 100%,
  stroke: 0.6pt + light-grey,
)


#let vueEnsemble(items) = box(
  width: 100%,
  fill: purple-light,
  inset: 12pt,
  radius: 4pt,
)[
  #for item in items {
    pad(left: 1em)[
       • #text(
        weight: "semibold",
        fill: purple,
      )[ #item.first() ]: #item.last()
    ]

  }
]

#let noteAnalyste(content) = {
  block(
    fill: notion.gray_bg,
    radius: 4pt,
    width: 100%,
    inset: 8pt,
    breakable: true,
  )[
    #pad(
      top: 6pt,
      bottom: 8pt
    )[
      #text(
        size: 14pt,
        weight: "extrabold",
        fill: purple-dark,
      )[Note de l'analyste]
      
      #content
    ]
  ]
}


#let article-indexing(article-index) = {
  for article in article-index {
    box(
      width: 100%,
      inset: (y: 10pt, x: 12pt),
      stroke: 0.5pt + light-grey,
      radius: 4pt,
      [
        #grid(
          columns: (0.6cm, auto, auto),
  
          [
            #text(
              weight: "bold",
              fill: accent,
            )[
              #article.first()
            ]
          ],
  
          [
            #text(
              weight: "bold",
              fill: dark,
            )[
              #article.at(1)
            ]
          ],
  
        )
      ]
    )
  
  }
}


#let article(
  category: "Article",
  number: "01",
  title: "",
  /*
  (
    ("jour mois année", [contenu], "URL"),
    ....
  )
  */
  events: (),
  /*
  overview: (
    ("Nom du bullet point (par exemple, Arsenal)", [contenu]),
  )
  */
  overview: (),
  body: [],
) = {
  pagebreak()

  text(
    size: 16pt,
    weight: "regular",
    fill: purple,
  )[
    #title
  ]

  grid(
    columns: (1fr, auto),
    align: (left, right),

    [
      #tag(category)
    ],

    [
      #text(
        size: 11pt,
        weight: "bold",
        fill: purple,
      )[
        #number
      ]
    ],
  )

  // Chronologie
  section-title[Chronologie]
  timeline(events)
  v(10pt)

  if overview != () [
    #block(
      sticky: true, 
      [
      #section-title[Vue d'ensemble]
      #vueEnsemble(overview)
      ])
    #v(10pt)
  ]
  
  
  // Synthèse
  section-title[Synthèse]

  set par(spacing: 2em)
  body
}






#let styled-table(columns, cell-align: auto, ..content) = {
  let cells = content.pos()

  let styled-cells = cells.enumerate().map(((i, cell)) => {
    let row = calc.floor(i / columns.len())

    if row == 0 {
      [
        #set text(
          weight: "bold",
          fill: white,
        )
        #cell
      ]
    } else {
      cell
    }
  })

  align(center, table(
    columns: columns,

    align: (x, y) => {
      if y == 0 {
        left
      } else {
        cell-align.at(x)
      }
    },

    fill: (_, y) => {
      if y == 0 {
        purple
      } else {
        none
      }
    },

    stroke: 1pt + gray,
    ..styled-cells,
  ))
}

#let fn(url) = {
  footnote[
    #better-link(url)[#url]
  ]
}

#let ioc(content) = {
  text(font: "Liberation Mono")[#content]
}

#let vt(body) = highlight( fill: luma(0%), radius: 2pt )[ #text( font: "Cascadia Mono", fill: white, size:11pt)[#body]]

#let ioc-list(
  title: [IOC],
  ips: [],
  domains: [],
  urls: [],
  files: [],
) = {
  set par(leading: 0pt)
  let parse-list(content) = {
    if "children" in content.fields() {
      content.children
    } else {
      (content,)
    }
  }

  let ioc-listt(items) = {
    
    for item in items {
      text(font: "Liberation Mono")[#item]
      linebreak()
    }
  }

  let ips = parse-list(ips)
  let domains = parse-list(domains)
  let urls = parse-list(urls)
  let files = parse-list(files)

  [
    #text(size: 13pt, fill:purple-dark)[*#title*]

    #if ips.len() > 0 [
      Adresses IP : \
      #ioc-listt(ips)
    ]

    #if domains.len() > 0 [
      Noms de domaine : \
      #ioc-listt(domains)
    ]

    #if urls.len() > 0 [
      URL : \
      #ioc-listt(urls)
    ]

    #if files.len() > 0 [
      Fichiers : \
      #ioc-listt(files)
    ]
  ]
}