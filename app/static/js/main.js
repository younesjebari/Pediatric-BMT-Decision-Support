// ===== Navigation Scroll Effect =====
window.addEventListener('scroll', function() {
    var navbar = document.getElementById('navbar');
    if (navbar) {
        if (window.scrollY > 20) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }
});

// ===== Mobile Nav Toggle =====
function toggleNav() {
    var links = document.getElementById('navLinks');
    if (links) {
        links.classList.toggle('open');
    }
}

// Close mobile nav on link click
document.querySelectorAll('.nav-links a').forEach(function(link) {
    link.addEventListener('click', function() {
        var links = document.getElementById('navLinks');
        if (links) links.classList.remove('open');
    });
});

// ===== Slider Value Updates =====
function updateSlider(input, valueId) {
    var val = parseFloat(input.value);
    document.getElementById(valueId).textContent = val % 1 === 0 ? val.toFixed(0) : val.toFixed(1);
}

// ===== Smooth Scroll for Anchor Links =====
document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
        var href = this.getAttribute('href');
        if (href === '#') return;
        var target = document.querySelector(href);
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// ===== Auto-dismiss Flash Messages =====
setTimeout(function() {
    document.querySelectorAll('.flash-msg').forEach(function(msg) {
        msg.style.transition = 'opacity .5s, transform .5s';
        msg.style.opacity = '0';
        msg.style.transform = 'translateY(-10px)';
        setTimeout(function() { msg.remove(); }, 500);
    });
}, 5000);

// ===== Intersection Observer for Animations =====
var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.feature-card, .card').forEach(function(el) {
    observer.observe(el);
});
