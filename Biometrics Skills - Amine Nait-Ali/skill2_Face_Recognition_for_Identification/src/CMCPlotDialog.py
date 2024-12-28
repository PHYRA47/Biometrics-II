from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QPushButton, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class CMCPlotDialog(QDialog):
    def __init__(self, current_embedding, database_embeddings, database_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CMC Curve")
        self.setGeometry(200, 200, 800, 600)
        
        # Store embeddings
        self.current_embedding = current_embedding
        self.database_embeddings = database_embeddings
        self.database_names = database_names
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Add close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
        
        # Plot CMC curve
        self.plot_cmc_curve()
    
    def calculate_cmc_data(self):
        # Calculate similarities between current face and all database faces
        similarities = []
        for db_embedding in self.database_embeddings:
            similarity = cosine_similarity(
                self.current_embedding.reshape(1, -1),
                db_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
        
        # Sort similarities in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Calculate CMC curve data
        n_samples = len(similarities)
        ranks = np.arange(1, n_samples + 1)
        cmc_curve = np.zeros(n_samples)
        
        # Find the rank of the correct match
        for i, rank in enumerate(ranks):
            # For demonstration, we'll consider matches above a threshold
            matches_at_rank = sorted_indices[:rank]
            cmc_curve[i] = len(matches_at_rank) / n_samples
        
        return ranks, cmc_curve
    
    def plot_cmc_curve(self):
        # Clear the figure
        self.figure.clear()
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Calculate and plot CMC curve
        ranks, cmc_curve = self.calculate_cmc_data()
        ax.plot(ranks, cmc_curve, 'b-', linewidth=2)
        
        # Customize plot
        ax.set_xlabel('Rank')
        ax.set_ylabel('Recognition Rate')
        ax.set_title('Cumulative Match Characteristic (CMC) Curve')
        ax.grid(True)
        
        # Set axis limits
        ax.set_xlim([1, len(ranks)])
        ax.set_ylim([0, 1.05])
        
        # Refresh canvas
        self.canvas.draw()