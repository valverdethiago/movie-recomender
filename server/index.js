const express = require('express');
const cors = require('cors');
const { users, movies } = require('./mockData'); // Import mock data

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

app.get('/api/users', (req, res) => res.json(users));

app.get('/api/movies', (req, res) => res.json(movies));

app.get('/api/movies/:id', (req, res) => {
  const movie = movies.find(m => m.id === req.params.id);
  if (!movie) return res.status(404).json({ error: 'Movie not found' });
  res.json(movie);
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});