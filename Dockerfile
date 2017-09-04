#import image cassandra and python2.7
FROM python:2.7-slim

# Set the working directory to /cassandra
WORKDIR /mnist-cassandra

# Copy the current directory contents into the container at /cassandra
ADD . /mnist-cassandra

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python","mnist-cassandra.py"]
