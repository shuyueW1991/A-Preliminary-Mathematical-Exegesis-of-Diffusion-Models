document.addEventListener('DOMContentLoaded', function() {
    // Disable right click
    document.addEventListener('contextmenu', function(e) {
        e.preventDefault();
    });
    
    // Disable text selection
    document.addEventListener('selectstart', function(e) {
        e.preventDefault();
    });
    
    // Disable copy shortcut
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && (e.keyCode === 67 || e.keyCode === 86 || e.keyCode === 85 || e.keyCode === 117)) {
            e.preventDefault();
        }
    });
    
    // Add elegant page turn effect
    document.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', function(e) {
            if (this.href && this.href.includes('.html') && !this.href.startsWith('http')) {
                e.preventDefault();
                document.body.style.opacity = '0';
                setTimeout(() => {
                    window.location.href = this.href;
                }, 300);
            }
        });
    });
    
    // Fade in content
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease';
        document.body.style.opacity = '1';
    }, 100);
});