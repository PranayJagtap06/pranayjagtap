document.addEventListener('DOMContentLoaded', function() {
    // Get the current page filename
    const currentPage = window.location.pathname.split("/").pop();

    // Get all nav links
    const navLinks = document.querySelectorAll('.nav-link');

    // Loop through the nav links
    navLinks.forEach(link => {
        // Get the href attribute
        const href = link.getAttribute('href');

        // Check if the href matches the current page
        if (href === currentPage) {
            link.classList.add('active'); // Add the active class
        }
    });
});

