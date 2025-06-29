---
layout: home
title: "Cover Page"
nav_order: 1

---
<!-- local side -->
<script>
  if (localStorage.getItem('visitCount')) {
    let count = parseInt(localStorage.getItem('visitCount')) + 1;
    localStorage.setItem('visitCount', count);
  } else {
    localStorage.setItem('visitCount', 1);
  }
  document.getElementById('visitor-counter').innerText = localStorage.getItem('visitCount');
</script>

<p>You’ve visited this page <span id="visitor-counter">0</span> times.</p>


<!-- statistics -->
<script>
  // Create a unique key for your page (e.g., based on page URL)
  const pageKey = "visits-" + window.location.pathname.replace(/\//g, '-');
  
  // Increment the counter and fetch the current count
  fetch(`https://api.countapi.xyz/hit/${pageKey}`)
    .then(response => response.json())
    .then(data => {
      document.getElementById('visitor-counter').innerText = data.value;
    });
</script>

<!-- Display the counter somewhere -->
<p>Visitors: <span id="visitor-counter">Loading...</span></p>




<div style="text-align: left; font-size: 1.8em;">
A Preliminary Mathematical Exegesis of Diffusion Model
<br> --  gaining clarity from true understanding
</div>



<br>

<div style="text-align: center; font-size: 1.4em;">
A PrgM2 (/pRˈɡem/2)'s work
</div>

<div style="text-align: center; font-size: 0.8em;">
The author reserves all right. No copy without the author's consent. No re-distribution without the author's consent. Let this thing just be minor contribution to our culture, a'ight? be cool.
</div>

<br>



<div style="text-align: center;">
  <img src="./assets/images/combined.png" style="width: 45%; max-width: 400px; height: auto; margin: 0 auto;">
</div>

<br>

<div style="text-align: left; font-size: 1.0em;">
[init]2025/06/16. Created vibe with jeykll and just-the-doc. Uploaded chapters 0, 1, and 2.
</div>

<div style="text-align: left; font-size: 1.0em;">
[grow]2025/06/18. Uploaded Chapter 3. Fixed some typos.
</div>

<div style="text-align: left; font-size: 1.0em;">
[grow]2025/06/18. Added aiding split line throughout the chapters. Fixed some typos.
</div>

<div style="text-align: left; font-size: 1.0em;">
[grow]2025/06/28. Updated chapters 0 and 1. Uploaded appendix for DDPM. Fixed some typos.
</div>

<div style="text-align: left; font-size: 1.0em;">
[grow]2025/06/29. Updated chapters 2, 3, and 4. 
</div>

<br>
<br>

