// Impide que la página se muestre embebida en un iframe ajeno.
//
// La cabecera correcta para esto (CSP frame-ancestors, o X-Frame-Options) no
// se puede enviar desde GitHub Pages, que no permite cabeceras propias, y
// tampoco funciona declarada en un <meta>. Así que se hace en JavaScript.
//
// Escapar del iframe puede fallar: si el atacante usa
// <iframe sandbox="allow-scripts"> sin allow-top-navigation, la asignación a
// window.top.location lanza SecurityError. Por eso lo primero es ocultar el
// documento y solo se vuelve a mostrar si resulta que no estábamos
// enmarcados. Enmarcado y en blanco es preferible a enmarcado y usable: el
// riesgo aquí no es robar una sesión (no hay), sino que alguien presente el
// veredicto del modelo dentro de una página que finja ser un portal clínico.
(function () {
  if (window.top === window.self) return;

  document.documentElement.style.display = 'none';
  try {
    window.top.location = window.self.location;
  } catch {
    // Navegación bloqueada por el sandbox del iframe: nos quedamos ocultos.
  }
})();
