from graphviz import Digraph

dot = Digraph(comment='UPI Fraud Detection Process')

# Nodes
dot.node('A', 'User Initiates UPI Transaction')
dot.node('B', 'Transaction Details\n(ID, Amount, Time, Location, Device ID)')
dot.node('C', 'Feature Extraction')
dot.node('D', 'ML Fraud Detection Model')
dot.node('E1', 'Legitimate Transaction')
dot.node('E2', 'Fraudulent Transaction\n(Alert/Block)')

# Edges
dot.edge('A', 'B', label='Enter Details')
dot.edge('B', 'C', label='Extract Features')
dot.edge('C', 'D', label='Analyze')
dot.edge('D', 'E1', label='Label: Legitimate')
dot.edge('D', 'E2', label='Label: Fraudulent')

# Save as PNG or SVG
dot.render('upi_fraud_detection_process', format='png', cleanup=True)
dot.render('upi_fraud_detection_process', format='svg', cleanup=True)
