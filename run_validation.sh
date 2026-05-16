#!/bin/bash
# Lance la validation complète sur Mac (test avant envoi Windows)

echo "═══════════════════════════════════════════"
echo "  PFE Smart Traffic — Validation SUMO"
echo "═══════════════════════════════════════════"

# Vérifier SUMO
if ! command -v netconvert &> /dev/null; then
    echo "❌ SUMO non trouvé. Installer: brew install sumo"
    exit 1
fi

# Générer réseau si absent
if [ ! -f "network/agadir.net.xml" ]; then
    echo "⚙️  Génération du réseau..."
    cd network
    netconvert \
        --node-files=agadir.nod.xml \
        --edge-files=agadir.edg.xml \
        --output-file=agadir.net.xml \
        --tls.guess true \
        --no-turnarounds true \
        --no-warnings
    cd ..
    echo "✓ Réseau généré"
fi

# Lancer validation
echo "🚦 Lancement validation MDP vs Fixe..."
python3 sumo_validator.py --mode compare

echo ""
echo "✓ Résultats → results/validation_results.json"