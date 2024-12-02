import shelve

# Save variables
with shelve.open('mydata') as db:
    db['x'] = 100
    db['list'] = [1, 2, 3]

# Load later
with shelve.open('mydata') as db:
    x = db['x']  # gets 100
    my_list = db['list']  # gets [1, 2, 3]