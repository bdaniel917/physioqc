"""Add keyboard annotations during signal replay and analysis."""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


def run(session: str, keys: str = "b:burst n:noise m:motion x:artifact") -> None:
    """Start annotation session for a processed signal.
    
    Args:
        session: Path to session directory containing processed results
        keys: Comma-separated key mappings (key:label)
    """
    # Parse key mappings
    key_mappings = {}
    for mapping in keys.split():
        if ':' in mapping:
            key, label = mapping.split(':', 1)
            key_mappings[key] = label
    
    print(f"Starting annotation session for: {session}")
    print(f"Available keys: {key_mappings}")
    print("Press 'q' to quit, 'h' for help")
    print("-" * 50)
    
    # Load session data
    try:
        # Load SQIs
        sqi_path = os.path.join(session, "sqi.parquet")
        if os.path.exists(sqi_path):
            import pandas as pd
            sqi_df = pd.read_parquet(sqi_path)
            print(f"Loaded {len(sqi_df)} SQI measurements")
        else:
            print("Warning: No SQI data found")
            sqi_df = None
        
        # Load existing events
        events_path = os.path.join(session, "events.jsonl")
        existing_events = []
        if os.path.exists(events_path):
            with open(events_path, 'r') as f:
                for line in f:
                    existing_events.append(json.loads(line.strip()))
            print(f"Loaded {len(existing_events)} existing events")
        
        # Load provenance
        provenance_path = os.path.join(session, "provenance.json")
        provenance = {}
        if os.path.exists(provenance_path):
            with open(provenance_path, 'r') as f:
                provenance = json.load(f)
        
    except Exception as e:
        print(f"Error loading session data: {e}")
        return
    
    # Start annotation loop
    annotations = []
    start_time = time.time()
    
    try:
        while True:
            # Get user input
            user_input = input("> ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 'h':
                print_help(key_mappings)
                continue
            elif user_input == 's':
                print_status(annotations, start_time, sqi_df)
                continue
            elif user_input == 'l':
                list_events(existing_events)
                continue
            elif user_input in key_mappings:
                # Add annotation
                timestamp = time.time() - start_time
                annotation = {
                    "timestamp": timestamp,
                    "annotation_type": "manual",
                    "label": key_mappings[user_input],
                    "key_pressed": user_input,
                    "session_time": datetime.now().isoformat(),
                }
                annotations.append(annotation)
                print(f"Added annotation: {key_mappings[user_input]} at {timestamp:.2f}s")
            else:
                print(f"Unknown key: {user_input}. Press 'h' for help.")
    
    except KeyboardInterrupt:
        print("\nAnnotation session interrupted.")
    
    # Save annotations
    if annotations:
        save_annotations(session, annotations, existing_events)
        print(f"Saved {len(annotations)} annotations")
    else:
        print("No annotations to save")


def print_help(key_mappings: Dict[str, str]) -> None:
    """Print help information."""
    print("\nAnnotation Help:")
    print("  Available keys:")
    for key, label in key_mappings.items():
        print(f"    {key}: {label}")
    print("  Commands:")
    print("    h: Show this help")
    print("    s: Show status")
    print("    l: List existing events")
    print("    q: Quit")
    print()


def print_status(annotations: List[Dict], start_time: float, sqi_df) -> None:
    """Print current session status."""
    elapsed = time.time() - start_time
    print(f"\nSession Status:")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"  Annotations: {len(annotations)}")
    
    if sqi_df is not None:
        print(f"  SQI measurements: {len(sqi_df)}")
        if len(sqi_df) > 0:
            latest_timestamp = sqi_df['timestamp'].max()
            print(f"  Latest SQI timestamp: {latest_timestamp:.2f}s")
    
    # Show recent annotations
    if annotations:
        print("  Recent annotations:")
        for ann in annotations[-5:]:
            print(f"    {ann['timestamp']:.2f}s: {ann['label']}")
    print()


def list_events(events: List[Dict]) -> None:
    """List existing events."""
    if not events:
        print("No existing events found.")
        return
    
    print(f"\nExisting Events ({len(events)}):")
    for i, event in enumerate(events[-10:]):  # Show last 10 events
        print(f"  {i+1}. {event['timestamp']:.2f}s: {event['event_type']} "
              f"(ch{event['channel']}, {event['severity']})")
    print()


def save_annotations(session: str, annotations: List[Dict], existing_events: List[Dict]) -> None:
    """Save annotations to session directory."""
    # Combine existing events with new annotations
    all_events = existing_events + annotations
    
    # Save to JSONL
    events_path = os.path.join(session, "events.jsonl")
    with open(events_path, 'w') as f:
        for event in all_events:
            f.write(json.dumps(event) + '\n')
    
    # Update provenance
    provenance_path = os.path.join(session, "provenance.json")
    if os.path.exists(provenance_path):
        with open(provenance_path, 'r') as f:
            provenance = json.load(f)
    else:
        provenance = {}
    
    provenance.update({
        "annotations_added": len(annotations),
        "total_events": len(all_events),
        "last_annotation_time": datetime.now().isoformat(),
    })
    
    with open(provenance_path, 'w') as f:
        json.dump(provenance, f, indent=2)
