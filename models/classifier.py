class MLPClassifier(nn.Module):
    def __init__(self, input_dim=384, output_dim=1):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)