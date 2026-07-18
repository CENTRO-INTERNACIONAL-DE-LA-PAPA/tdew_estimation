// Preámbulo de estilo para la salida Typst (se inyecta vía `include-in-header`).
// No fija página/fuente/tamaño: eso lo controla el YAML de Quarto
// (papersize, margin, fontsize, mainfont) para evitar conflictos.

// Párrafos justificados con guionado.
#set par(justify: true)
#set text(hyphenate: true)

// Encabezados con color de acento (azul corporativo sobrio).
#show heading.where(level: 1): set text(fill: rgb("#1F4E79"))
#show heading.where(level: 2): set text(fill: rgb("#2E74B5"))
#show heading.where(level: 3): set text(fill: rgb("#595959"))

// Bloques de código / pseudocódigo: fondo suave, esquinas redondeadas
// y una barra de acento a la izquierda.
#show raw.where(block: true): it => block(
  fill: rgb("#F4F6F8"),
  inset: 9pt,
  radius: 3pt,
  width: 100%,
  stroke: (left: 2pt + rgb("#2E74B5")),
  text(size: 8.5pt, it),
)

// Enlaces (DOIs, URLs) en color de acento.
#show link: set text(fill: rgb("#2E74B5"))
