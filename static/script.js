const cards = document.querySelectorAll(".card-details")

cards.forEach(card => {
    card.addEventListener('click', function(){
        const title = this.querySelector('h2').textContent; 
        const location = this.querySelector('#city_name').textContent; 
        const searchQuery = "Directions to " + title + " " + location.split(":")[1].trim(); 
        window.open(`https://www.google.com/search?q=${searchQuery}`); 
    }); 

}); 
