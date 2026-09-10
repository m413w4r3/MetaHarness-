// CONFIGURATION

// #set document(
//   title: "Rapport ABC",
//   author: "X",
// )

#set text(
  font: "Hanken Grotesk",
  size: 11pt,
  lang: "fr",
)

#set page(
  paper: "a4",
  margin: (
    top: 2.3cm,
    bottom: 2.3cm,
    left: 2.2cm,
    right: 2.2cm,
  ),
)

// Code styling
// #show raw.where(block: false): it => box(inset: (x: 3pt), outset: (y: 3pt), radius: 40%, fill: luma(170), it)
// #show raw.where(block: false): box.with(
//     fill: luma(190),
//     inset: (x: 3pt, y: 0pt),
//     outset: (y: 3pt),
//     radius: 2pt,
// )
#show raw.where(block: false): highlight.with( fill: luma(190),  radius: 2pt)
#show raw.where(block: false): set text(size: 10pt)

#show raw.where(block: true): it => pad(x: 2%, block(
    width: 100%,
    fill: black,
    inset: (x: 10pt, y: 1pt),
    outset: (y: 5pt),
    radius: 2pt,
  )[
    #set text(fill: white)
    #it
  ],
)

#show raw: set text(font: "Cascadia Mono")
#set par(justify: true)

// Gray background, with superscript number footnote
#show footnote: it => {
    set text(weight: "regular", size: 12pt)
    highlight(fill: gray.transparentize(20%), extent: 1pt, [#super[#counter(footnote).at(it.location()).first()]]) 
}
#set footnote.entry(indent: 0em)


#set figure(supplement: [Figure])
#set figure.caption(separator: [ : ])
#show figure.caption: set text(9pt)
#show figure.caption: emph

#set enum(indent: 2em)
#set list(indent: 2em)


// Couleurs
#import "UTILS/colors.typ": *

#import "UTILS/header_footer.typ": *
// Appliquer header/footer sur toutes les pages
#set page(
  header: report-header,
  footer: report-footer,
)

// Styles de titres
#show heading.where(level: 1): it => block(
  above: 28pt,
  below: 16pt,
  [
    #text(
      size: 20pt,
      weight: "extrabold",
      fill: purple,
    )[ #it.body ]
  ],
)

#show heading.where(level: 2): it => block(
  above: 20pt,
  below: 12pt,
  [
    #text(
      size: 14pt,
      weight: "bold",
      fill: purple,
    )[ #it.body ]
  ],
)

#show heading.where(level: 3): it => block(
  above: 15pt,
  below: 8pt,
  [
    #text(
      size: 12pt,
      weight: "bold",
      fill: purple,
    )[ #it.body ]
  ],
)

#show heading.where(level: 4): it => block(
  above: 15pt,
  below: 8pt,
  [
    #text(
      size: 10pt,
      weight: "bold",
      fill: purple,
    )[ #it.body ]
  ],
)




#import "UTILS/helpers.typ": tag, section-title, timeline, separator, vueEnsemble, noteAnalyste, better-link, article-indexing, article, styled-table

// PAGE DE GARDE


// = Actualité des codes et infrastructures X
#align(center)[
  #text(
    size: 20pt,
    weight: "extrabold",  
    fill: purple,
  )[Actualité des codes et infrastructures X]
]

#v(16pt)

== Résumé de l'actualité #tag("Articles")


// PAGE DE GARDE ARTICLE

#let article-index = (
  ("01", better-link(<article01>)[[Groupe] Titre]),
  ("02", better-link(<article02>)[[Groupe] Titre]),
)

#article-indexing(article-index)

#v(16pt)

== Autres brèves relevées #tag("Brèves")


// PAGE DE GARDE BREVE
#v(8pt)

#let breve-index = (
  ("03", better-link(<breve01>)[Titre 1]),
  ("04", better-link(<breve02>)[Titre 2]),
)

#article-indexing(breve-index)

#v(16pt)

== Règles Suricata _"Emerging Threats"_
#v(8pt)
- Règles modifiées : ?

#v(16pt)
== Suivi des indicateurs
- Échantillons relevés : ?
    
#v(16pt)

#styled-table(
  (1fr, 4fr),
  cell-align: (center, left),

  [Fichier intégré], [Description],
  [📎 ?], [Données CTI issues du bulletin, au format STIX.],
  [📎 ?], [Synthèse des thèmes et des IOC collectés, sous forme d’un tableau Excel.],
  [📎 ?], [Annexes associées au bulletin, sous forme d’une archive ZIP ; le fichier `.txt` doit être renommé avec une extension `.zip` avant extraction.],
)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ARTICLES
#include "articles/article01/article01.typ"
<article01> 

#include "articles/article02/article02.typ"
<article02> 


// BREVES
#include "breves/breve01/breves01.typ"
<breve01>

#include "breves/breve02/breve02.typ"
<breve02>

#pagebreak()
#include "articles_non_traites.typ"