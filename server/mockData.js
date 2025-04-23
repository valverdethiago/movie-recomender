const users = [
    {id: '1', name: 'Alice'},
    {id: '2', name: 'Bob'},
    {id: '3', name: 'Charlie'},
    {id: '4', name: 'David'},
    {id: '5', name: 'Eve'}
];

const movies = Array.from({length: 40}, (_, i) => ({
    id: (i + 1).toString(),
    title: `Movie ${i + 1}`,
    imageUrl: `https://picsum.photos/500/300?random=${i + 1}`,
    description: `This is the description for Movie ${i + 1}. A great film to watch!`,
    userId: users[i % users.length].id // Distribute movies among users
}));

module.exports = {users, movies};