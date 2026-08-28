document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.wy-menu-vertical p.caption').forEach(function (caption) {
        const span = caption.querySelector('.caption-text');
        if (!span) return;

        const ul = caption.nextElementSibling;
        if (!ul || ul.tagName !== 'UL') return;
        const firstLink = ul.querySelector('a');
        if (!firstLink) return;

        // Construct index.html path from the first item's href (same directory)
        const href = firstLink.getAttribute('href');
        const indexHref = href.replace(/[^/]+\.html(#.*)?$/, 'index.html');
        if (indexHref === href) return; // First item is already an index — caption is redundant

        // Wrap the entire <p> in an <a>
        const a = document.createElement('a');
        a.href = indexHref;
        a.className = 'caption-link';
        caption.parentNode.insertBefore(a, caption);
        a.appendChild(caption);
    });
});
