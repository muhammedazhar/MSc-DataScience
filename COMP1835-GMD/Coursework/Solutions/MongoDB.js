db.Books.insertMany([
  {
    title: "The Great Gatsby",
    author: "F. Scott Fitzgerald",
    publishedYear: 1925,
    genres: ["Novel", "Fiction"],
    reviews: [{ user: "Reader123", rating: 5, comment: "A timeless masterpiece." }],
    availability: { online: true, stores: ["New York", "Los Angeles"] }
  },
  {
    title: "1984",
    author: "George Orwell",
    publishedYear: 1949,
    genres: ["Science Fiction", "Dystopian"],
    availability: { online: true, stores: [] }
  },
  {
    title: "To Kill a Mockingbird",
    author: "Harper Lee",
    publishedYear: 1960,
    genres: ["Novel", "Southern Gothic"],
    availability: { online: true, stores: ["Chicago", "San Francisco"] }
  },
  {
    title: "Harry Potter and the Philosopher's Stone",
    author: "J.K. Rowling",
    publishedYear: 1997,
    genres: ["Fantasy", "Young Adult"],
    availability: { online: true, stores: ["London", "Paris"] }
  },
  {
    title: "Pride and Prejudice",
    author: "Jane Austen",
    publishedYear: 1813,
    genres: ["Romance", "Classic"],
    availability: { online: true, stores: ["London"] }
  },
  {
    title: "The Catcher in the Rye",
    author: "J.D. Salinger",
    publishedYear: 1951,
    genres: ["Novel", "Coming-of-age"],
    availability: { online: false, stores: [] }
  },
  {
    title: "Moby-Dick",
    author: "Herman Melville",
    publishedYear: 1851,
    genres: ["Adventure", "Epic"],
    availability: { online: false, stores: [] },
    themes: ["Whaling", "Obsession"]
  },
  {
    title: "The Hobbit",
    author: "J.R.R. Tolkien",
    publishedYear: 1937,
    genres: ["Fantasy", "Adventure"],
    availability: { online: true, stores: ["Middle Earth"] },
    characters: ["Bilbo Baggins", "Gandalf", "Thorin Oakenshield"]
  },
  {
    title: "The Lord of the Rings: The Fellowship of the Ring",
    author: "J.R.R. Tolkien",
    publishedYear: 1954,
    genres: ["Fantasy", "Adventure"],
    availability: { online: true, stores: ["Middle Earth"] },
    characters: ["Frodo Baggins", "Samwise Gamgee", "Aragorn"]
  },
  {
    title: "The Hitchhiker's Guide to the Galaxy",
    author: "Douglas Adams",
    publishedYear: 1979,
    genres: ["Science Fiction", "Comedy"],
    availability: { online: true, stores: ["Galaxy", "Earth"] },
    characters: ["Arthur Dent", "Ford Prefect", "Zaphod Beeblebrox"]
  },
  {
    title: "The Alchemist",
    author: "Paulo Coelho",
    publishedYear: 1988,
    genres: ["Fiction", "Philosophical"],
    availability: { online: true, stores: ["Spain", "Brazil"] }
  },
  {
    title: "The Da Vinci Code",
    author: "Dan Brown",
    publishedYear: 2003,
    genres: ["Mystery", "Thriller"],
    availability: { online: true, stores: ["Paris", "Rome"] }
  },
  {
    title: "The Hunger Games",
    author: "Suzanne Collins",
    publishedYear: 2008,
    genres: ["Dystopian", "Young Adult"],
    availability: { online: true, stores: ["District 12", "Capital City"] }
  },
  {
    title: "The Road",
    author: "Cormac McCarthy",
    publishedYear: 2006,
    genres: ["Post-Apocalyptic", "Dystopian"],
    availability: { online: true, stores: ["United States"] }
  },
  {
    title: "The Picture of Dorian Gray",
    author: "Oscar Wilde",
    publishedYear: 1890,
    genres: ["Gothic Fiction", "Philosophical"],
    availability: { online: true, stores: ["London", "Paris"] }
  },
  {
    title: "The Road Not Taken",
    author: "Robert Frost",
    publishedYear: 1916,
    genres: ["Poetry"],
    availability: { online: true, stores: [] }
  },
  {
    title: "The Shining",
    author: "Stephen King",
    publishedYear: 1977,
    genres: ["Horror", "Psychological Fiction"],
    availability: { online: true, stores: ["Overlook Hotel", "Colorado"] }
  },
  {
    title: "The Girl with the Dragon Tattoo",
    author: "Stieg Larsson",
    publishedYear: 2005,
    genres: ["Mystery", "Thriller"],
    availability: { online: true, stores: ["Stockholm"] }
  },
  {
    title: "The Chronicles of Narnia: The Lion, the Witch and the Wardrobe",
    author: "C.S. Lewis",
    publishedYear: 1950,
    genres: ["Fantasy", "Children's Literature"],
    availability: { online: true, stores: ["Narnia"] },
    characters: ["Aslan", "Lucy Pevensie", "Edmund Pevensie"]
  },
  {
    title: "The War of the Worlds",
    author: "H.G. Wells",
    publishedYear: 1898,
    genres: ["Science Fiction"],
    availability: { online: true, stores: ["England"] },
    adaptations: ["Radio Drama", "Film", "TV Series"]
  },
  {
    title: "Frankenstein",
    author: "Mary Shelley",
    publishedYear: 1818,
    genres: ["Gothic", "Science Fiction"],
    availability: { online: true, stores: ["Geneva", "Ingolstadt"] }
  },
  {
    title: "The Handmaid's Tale",
    author: "Margaret Atwood",
    publishedYear: 1985,
    genres: ["Dystopian", "Feminist Literature"],
    availability: { online: true, stores: ["Republic of Gilead"] }
  },
  {
    title: "The Adventures of Sherlock Holmes",
    author: "Arthur Conan Doyle",
    publishedYear: 1892,
    genres: ["Mystery", "Detective Fiction"],
    availability: { online: true, stores: ["London"] },
    characters: ["Sherlock Holmes", "Dr. John Watson", "Professor Moriarty"]
  },
  {
    title: "Alice's Adventures in Wonderland",
    author: "Lewis Carroll",
    publishedYear: 1865,
    genres: ["Fantasy", "Children's Literature"],
    availability: { online: true, stores: ["Wonderland"] },
    characters: ["Alice", "The White Rabbit", "The Mad Hatter"]
  },
  {
    title: "Dracula",
    author: "Bram Stoker",
    publishedYear: 1897,
    genres: ["Gothic Horror", "Epistolary Novel"],
    availability: { online: true, stores: ["Transylvania", "London"] }
  },
  {
    title: "The Catcher in the Rye",
    author: "J.D. Salinger",
    publishedYear: 1951,
    genres: ["Novel", "Coming-of-age"],
    availability: { online: false, stores: [] }
  },
  {
    title: "The Old Man and the Sea",
    author: "Ernest Hemingway",
    publishedYear: 1952,
    genres: ["Novella"],
    availability: { online: true, stores: ["Havana", "Cuba"] }
  },
  {
    title: "The Metamorphosis",
    author: "Franz Kafka",
    publishedYear: 1915,
    genres: ["Absurdist Fiction", "Surrealism"],
    availability: { online: true, stores: ["Prague"] }
  },
  {
    title: "One Hundred Years of Solitude",
    author: "Gabriel García Márquez",
    publishedYear: 1967,
    genres: ["Magical Realism", "Literary Fiction"],
    availability: { online: true, stores: ["Macondo"] }
  },
  {
    title: "The Count of Monte Cristo",
    author: "Alexandre Dumas",
    publishedYear: 1844,
    genres: ["Adventure", "Historical Fiction"],
    availability: { online: true, stores: ["France"] }
  }
]);