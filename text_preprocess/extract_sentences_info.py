import json
import re

def extract_info(sentence, instance_label, class_mapping, variant_mapping, modality_keywords, direction_patterns):
    """
    Extract modality and specified class (anatomy or lesion) ID and name from the sentence.
    Prioritize longer and more specific terms to avoid partial matches (e.g., 'Brainstem' over 'Brain').
    Handle term variants (e.g., 'pancreatic' for 'Pancreas') using variant_mapping.
    Args:
        sentence (str): Input sentence, e.g., "Brainstem in abdominal CT"
        instance_label (int): 0 for anatomy, 1 for lesion
    Returns:
        dict: Contains modality and specified class ID and name
    """
    result = {
        'modality': None,
        'class': {'id': None, 'name': None}
    }

    # Extract modality
    for modality, keywords in modality_keywords.items():
        if any(keyword in sentence for keyword in keywords):
            result['modality'] = modality
            break

    sentence_lower = sentence.lower()

    # If modality is found, look up anatomy or lesion based on instance_label
    if result['modality']:
        modality_data = class_mapping.get(result['modality'], {})
        class_dict = modality_data.get('anatomy' if instance_label == 0 else 'lesion', {})
        
        # Create combined dictionary of class names and their variants
        combined_terms = {}
        
        # Add original class names with their length as part of the value
        for class_name, class_id in class_dict.items():
            combined_terms[class_name] = {
                'id': class_id,
                'canonical_name': class_name,
                'length': len(class_name)
            }
            
        # Add variant terms that exist in the class_dict, with their length
        for variant, original_name in variant_mapping.items():
            if original_name in class_dict:
                combined_terms[variant] = {
                    'id': class_dict[original_name],
                    'canonical_name': original_name,
                    'length': len(variant)
                }

        # Convert to list and sort by length (descending) in one step
        # Using a tuple (-length, term) to ensure stable sorting
        sorted_terms = sorted(
            combined_terms.items(),
            key=lambda x: (-x[1]['length'], x[0])
        )

        detected_direction = None
        for direction, pattern in direction_patterns.items():
            if pattern.search(sentence_lower):
                detected_direction = direction.capitalize()
                break

        # Check for matches in the sorted terms
        for term, term_data in sorted_terms:
            if detected_direction is not None and detected_direction.lower() in term.lower():
                no_direction_term = term.lower().replace(detected_direction.lower(), "").strip()
                if re.search(r'\b' + re.escape(no_direction_term.lower()) + r'\b', sentence_lower):
                    result['class']['id'] = term_data['id']
                    result['class']['name'] = term_data['canonical_name']
                    return result  # Return early if match is found
            else:
                if re.search(r'\b' + re.escape(term.lower()) + r'\b', sentence_lower):
                    result['class']['id'] = term_data['id']
                    result['class']['name'] = term_data['canonical_name']
                    return result  # Return early if match is found

    return result

if __name__ == '__main__':
    # Load class_mapping.json
    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)

    with open('variant_mapping.json', 'r') as f:
        variant_mapping = json.load(f)

    # Define modality keywords
    modality_keywords = {
        'Microscopy': ['microscopy', 'microscope', 'Microscope', 'Microscopy', 'microscopic', 'Microscopic', 'microscopical', 'Microscopical', 'ultrastructural', 'Ultrastructural', 'ultrastructure', 'Ultrastructure', 'EM', 'light sheet', 'Light sheet'],
        'PET': ['PET', 'positron emission tomography'],
        'US': ['US', 'ultrasound', 'Ultrasound', 'Echocardiography', 'echocardiography', 'Echocardiographic', 'echocardiographic', 'ultrasonic', 'Ultrasonic'],
        'MRI': ['MR', 'MRI', 'magnetic resonance', 'Magnetic resonance', 'Magnetic Resonance', 'diffusion', 'Diffusion', 'DWI', 'ADC', 'pelvic'],
        'CT': ['CT', 'computed tomography', 'Computed tomography', 'Computed Tomography', 'tomographic', 'Tomographic', 'cross-sectional']
    }
    direction_patterns = {
        'left': re.compile(r'\b(left)\b', re.IGNORECASE),
        'right': re.compile(r'\b(right)\b', re.IGNORECASE)
    }

    # Test
    sentence = "Brainstem in CT"
    instance_label = 0  # 0 for anatomy
    result = extract_info(sentence, instance_label, class_mapping, variant_mapping, modality_keywords, direction_patterns)
    print(result)

    sentence = "Brain in CT"
    instance_label = 0  # 0 for anatomy
    result = extract_info(sentence, instance_label, class_mapping, variant_mapping, modality_keywords, direction_patterns)
    print(result)

    sentence = "Liver in CT"
    instance_label = 0  # 0 for anatomy
    result = extract_info(sentence, instance_label, class_mapping, variant_mapping, modality_keywords, direction_patterns)
    print(result)
    
    # Another test case
    sentence = "Liver tumors in abdominal CT"
    instance_label = 1  # 1 for lesion
    result = extract_info(sentence, instance_label, class_mapping, variant_mapping, modality_keywords, direction_patterns)
    print(result)