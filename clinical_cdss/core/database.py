from neo4j import GraphDatabase

from clinical_cdss.core import config


class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
        )

    def close(self):
        if self.driver:
            self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def init_constraints(self):
        constraints = [
            "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT evidence_case_id IF NOT EXISTS FOR (ec:EvidenceCase) REQUIRE ec.id IS UNIQUE",
            "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT diagnostic_rule_name IF NOT EXISTS FOR (r:Diagnostic_Rule) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT guideline_chunk_id IF NOT EXISTS FOR (g:GuidelineChunk) REQUIRE g.id IS UNIQUE",
        ]
        for query in constraints:
            self.execute_query(query)

    def init_vector_index(self):
        indexes = [
            """
            CREATE VECTOR INDEX symptom_embedding_index IF NOT EXISTS
            FOR (s:Symptom)
            ON (s.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 768,
             `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX guideline_chunk_index IF NOT EXISTS
            FOR (g:GuidelineChunk)
            ON (g.embedding)
            OPTIONS {indexConfig: {
             `vector.dimensions`: 768,
             `vector.similarity_function`: 'cosine'
            }}
            """,
        ]
        for query in indexes:
            self.execute_query(query)
        print("Initialized Neo4j vector indexes.")

    def init_schema(self):
        self.init_constraints()
        self.init_vector_index()
