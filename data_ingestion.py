import os
from dotenv import load_dotenv
import json
from pinecone import Pinecone,ServerlessSpec
from sentence_transformers import SentenceTransformer
import torch
from huggingface_hub import InferenceClient

load_dotenv(override=True)

hf_token = os.getenv("HUGGINGFACE_API_KEY")
if not hf_token:
    raise ValueError("Hugging Face API token not found. Please set the HF_TOKEN environment variable.")
client = InferenceClient(api_key=hf_token)

index_name = "flight-chatbot-rag"

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),  # Use the API key directly
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(index_name)


model = SentenceTransformer('intfloat/multilingual-e5-large')

def generate_embedding(text):
    if not isinstance(text, str):
        text = str(text)
    with torch.no_grad():
        embeddings = model.encode(text)
    return embeddings.tolist()

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# def process_and_store_data(data):
#     user_id = data['user']['id']
    
#     for flight in data['user']['flights']:
#         ticket_id = flight['ticket_id']
        
#         journey_text = f"Flight from {flight['source']} to {flight['destination']} departs on {flight['departure_date']} and arrives on {flight['arrival_date']} with a layover of {flight['layover_duration']}."
#         embedding = generate_embedding(journey_text)
#         index.upsert([(str(ticket_id), embedding, {"text": journey_text})])
        
#         for segment in flight['segments']:
#             segment_text = f"{segment['flight_number']} from {segment['departure']['airport']} ({segment['departure']['iata']}) to {segment['arrival']['airport']} ({segment['arrival']['iata']}) departs on {segment['departure']['date']} and arrives on {segment['arrival']['date']}."
#             embedding = generate_embedding(segment_text)
#             index.upsert([(f"{ticket_id}-{segment['flight_number']}", embedding, {"text": segment_text})])
            
#             for passenger in segment['passengers']:
#                 passenger_text = f"{passenger['first_name']} {passenger['last_name']} is seated in {passenger['seat_number']} with {passenger['cabin_baggage']} cabin baggage and {passenger['check_in_baggage']} checked baggage."
#                 embedding = generate_embedding(passenger_text)
#                 index.upsert([(f"{ticket_id}-{segment['flight_number']}-{passenger['seat_number']}", embedding, {"text": passenger_text})])

def process_and_store_data(data):
    user = data.get('user')
    if not user:
        print("No user data found.")
        return
    
    user_id = user.get('id')
    if not user_id:
        print("User ID is missing.")
        return
    
    flights = user.get('flights', [])
    for flight in flights:
        ticket_id = flight.get('ticket_id')
        if not ticket_id:
            print("Ticket ID is missing for a flight.")
            continue  # Skip this flight if ticket_id is missing
        
        # Process the journey
        journey_text = (
            f"Flight from {flight['source']} to {flight['destination']} departs on "
            f"{flight['departure_date']} and arrives on {flight['arrival_date']} with "
            f"a layover of {flight['layover_duration']}."
        )
        
        embedding = generate_embedding(journey_text)
        
        # Build metadata for the journey
        journey_metadata = {
            "user_id": user_id,
            "ticket_id": ticket_id,
            "source": flight['source'],
            "destination": flight['destination'],
            "departure_date": flight['departure_date'],
            "arrival_date": flight['arrival_date'],
            "layover_duration": flight['layover_duration'],
            "class": flight.get('class', 'N/A'),
            "pnr": flight.get('pnr', 'N/A'),
            "text": journey_text
        }
        
        # Upsert journey into index with more metadata
        index.upsert([
            (
                f"{user_id}-{ticket_id}",
                embedding,
                journey_metadata
            )
        ])
        
        # Process segments
        segments = flight.get('segments', [])
        for segment in segments:
            flight_number = segment.get('flight_number')
            if not flight_number:
                print("Flight number is missing for a segment.")
                continue  # Skip this segment if flight_number is missing
            
            segment_text = (
                f"Flight {flight_number} from {segment['departure']['airport']} "
                f"({segment['departure']['iata']}) to {segment['arrival']['airport']} "
                f"({segment['arrival']['iata']}) departs on {segment['departure']['date']} "
                f"and arrives on {segment['arrival']['date']}."
            )
            
            embedding = generate_embedding(segment_text)
            
            # Build metadata for the segment
            segment_metadata = {
                "user_id": user_id,
                "ticket_id": ticket_id,
                "flight_number": flight_number,
                "departure_airport": segment['departure']['airport'],
                "departure_iata": segment['departure']['iata'],
                "departure_date": segment['departure']['date'],
                "arrival_airport": segment['arrival']['airport'],
                "arrival_iata": segment['arrival']['iata'],
                "arrival_date": segment['arrival']['date'],
                "text": segment_text
            }
            
            # Upsert segment into index with more metadata
            index.upsert([
                (
                    f"{user_id}-{ticket_id}-{flight_number}",
                    embedding,
                    segment_metadata
                )
            ])
            
            # Process passengers
            passengers = segment.get('passengers', [])
            for passenger in passengers:
                seat_number = passenger.get('seat_number')
                first_name = passenger.get('first_name')
                last_name = passenger.get('last_name')
                if not seat_number or not first_name or not last_name:
                    print("Passenger information is incomplete.")
                    continue  # Skip this passenger if essential info is missing
                
                passenger_text = (
                    f"{first_name} {last_name} is seated in {seat_number} with "
                    f"{passenger['cabin_baggage']} cabin baggage and "
                    f"{passenger['check_in_baggage']} checked baggage."
                )
                
                embedding = generate_embedding(passenger_text)
                
                # Build metadata for the passenger
                passenger_metadata = {
                    "user_id": user_id,
                    "ticket_id": ticket_id,
                    "flight_number": flight_number,
                    "seat_number": seat_number,
                    "first_name": first_name,
                    "last_name": last_name,
                    "cabin_baggage": passenger.get('cabin_baggage', 'N/A'),
                    "check_in_baggage": passenger.get('check_in_baggage', 'N/A'),
                    "text": passenger_text
                }
                
                # Upsert passenger into index with more metadata
                index.upsert([
                    (
                        f"{user_id}-{ticket_id}-{flight_number}-{seat_number}",
                        embedding,
                        passenger_metadata
                    )
                ])
                # print(f"Upserted Passenger ID: {user_id}-{ticket_id}-{flight_number}-{seat_number}")
                # print(f"Passenger Name: {first_name} {last_name}")
                # print(f"Metadata: {passenger_metadata}")
                # print("-----")

def query_and_generate_response(query, user_id):
    query_embedding = generate_embedding(query)
    
    # Simplified metadata filter
    metadata_filter = {
        "user_id": user_id
    }
    
    # Query the index using the simplified filter
    results = index.query(
        vector=query_embedding, 
        top_k=10, 
        include_metadata=True,
        include_values=False,
        include_scores=True,
        filter=metadata_filter
    )
    
    # Check if any matches are returned
    matches = results.get('matches', [])
        # Debugging: Print out matches and scores
    print("Query:", query)
    for match in matches:
        print(f"Score: {match['score']}, Text: {match['metadata']['text']}")
    if matches:
        # Adjust the threshold and comparison based on your vector database's scoring
        threshold = 0.77  # Adjust as necessary
        # Assuming higher scores mean more similarity
        relevant_matches = [
            match for match in matches 
            if match['score'] >= threshold
        ]
    else:
        relevant_matches = []
    
    if relevant_matches:
        # Construct retrieved information
        retrieved_info = "\n".join([
            match['metadata']['text'] for match in relevant_matches
        ])
        prompt = f'''
User Query: {query}
Relevant Information:{retrieved_info}
Assist customers traveling by flight by providing relevant information as needed.
Ensure to address their queries promptly and concisely. 
Give information for all the passenger.
STRICT:Use all the information provided by the relevant information and return what is asked.
Provide a user-friendly response based on the above information with short words.
        '''
    else:
        prompt = f'''
User Query: {query}
Assist customers traveling by flight by providing relevant information as needed. 
Provide a user-friendly response.
        '''
        
    print("Query:", query)
    print("Prompt:", prompt)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct", 
        messages=messages, 
        temperature=0,
        max_tokens=500,
    )
    return response.choices[0].message["content"]


data = load_data(r"D:\company\hava havai\flight-chatbot-with-RAG\Data\Journey_Details.json")
process_and_store_data(data)
print("Data ingested successfully!")