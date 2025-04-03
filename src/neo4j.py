from neo4j import GraphDatabase
from dotenv import load_dotenv
import json
import os

load_dotenv()

class Neo4jConnector:
    def __init__(self, uri, user, password):
        # Initialize connection to Neo4j database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # Close the database connection
        self.driver.close()

    def load_json(self, filepath):
        # Load the JSON file and extract nodes and edges
        with open(filepath, "r") as file:
            data = json.load(file)
        return data['elements']['nodes'], data['elements']['edges']

    def create_graph(self, nodes, edges):
        # Process all nodes and edges using transactions
        with self.driver.session() as session:
            for node in nodes:
                person_data = node['data']
                session.execute_write(self._create_person_and_attributes, person_data)

            for edge in edges:
                edge_data = edge['data']
                session.execute_write(self._create_connection, edge_data)

    @staticmethod
    def _create_person_and_attributes(tx, data):
        person_id = str(data["id"])

        # Basic identity properties for the Person node
        person_core = {
            "id": person_id,
            "name": str(data["name"]),
            "value": str(data["value"]),
            "first_name": str(data["first_name"]),
            "last_name": str(data["last_name"])
        }

        # Create or update the Person node
        tx.run("""
            MERGE (p:Person {id: $id})
            SET p += $props
        """, id=person_id, props=person_core)

        # Define generic attributes to be stored as individual nodes
        attribute_map = {
            "designation": "Designation",
            "department": "Department",
            "location": "Location",
            "reporting_manager": "ReportingManager",
            "joining_date": "JoiningDate",
            "email": "Email",
            "legal_entity": "LegalEntity",
            "group_name1": "GroupName1",
            "group_name2": "GroupName2",
            "group_name3": "GroupName3",
            "group_name4": "GroupName4",
            "group_name5": "GroupName5",
            "rating": "Rating"
        }

        # Create each attribute node and link it to the person
        for attr, label in attribute_map.items():
            value = data.get(attr)
            if value not in [None, "null", "None"]:
                tx.run(f"""
                    MERGE (a:{label} {{value: $value}})
                    MERGE (p:Person {{id: $person_id}})
                    MERGE (p)-[:HAS_{label.upper()}]->(a)
                """, value=value, person_id=person_id)

        # Handle Gender separately — use :Gender node with `type` instead of `value`
        gender = data.get("gender")
        if gender not in [None, "null", "None"]:
            tx.run("""
                MERGE (g:Gender {type: $gender})
                MERGE (p:Person {id: $person_id})
                MERGE (p)-[:HAS_GENDER]->(g)
            """, gender=gender, person_id=person_id)

        # Handle HierarchyLevel separately — use :HierarchyLevel node with `level`
        hl = data.get("hierarchy_level")
        if hl is not None:
            tx.run("""
                MERGE (hl:HierarchyLevel {level: $hl})
                MERGE (p:Person {id: $person_id})
                MERGE (p)-[:HAS_HIERARCHY_LEVEL]->(hl)
            """, hl=int(hl), person_id=person_id)

        # Handle Leadership separately — use :Leadership node with `type`
        leadership = data.get("leadership")
        if leadership not in [None, "null", "None"]:
            tx.run("""
                MERGE (l:Leadership {type: $leadership})
                MERGE (p:Person {id: $person_id})
                MERGE (p)-[:HAS_LEADERSHIP]->(l)
            """, leadership=leadership, person_id=person_id)

    @staticmethod
    def _create_connection(tx, edge):
        # Create directed IS_CONNECTED relationship between Person nodes
        source = str(edge["source"])
        target = str(edge["target"])

        tx.run("""
            MATCH (src:Person {id: $source}), (tgt:Person {id: $target})
            MERGE (src)-[:IS_CONNECTED]->(tgt)
        """, source=source, target=target)

if __name__ == "__main__":
    connector = Neo4jConnector(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    # Load JSON data
    nodes, edges = connector.load_json("data/cyto.json")
    
    # Create graph in Neo4j
    connector.create_graph(nodes, edges)
    
    # Close the connection
    connector.close()
    print("Graph created successfully in Neo4j database.")