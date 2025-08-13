import os
import json
from socket import if_nameindex
from datasets.panoptic_dataset import PanopticDataset


class CaptionPanopticDataset(PanopticDataset):
    """Extended PanopticDataset that can load captions from JSON files"""

    def __init__(self, name, image_dir, ann_file, mask_dir, caption_dir=None):
        super().__init__(name, image_dir, ann_file, mask_dir)
        self.caption_dir = caption_dir
        self.captions = {}
        self.has_captions = False
        self.caption_load_summary = {}

    def load_captions(self):
        """Load captions from caption directory with strict validation and reporting"""
        if not self.caption_dir or not os.path.exists(self.caption_dir):
            print(f"No caption directory provided or directory does not exist: {self.caption_dir}")
            self.has_captions = False
            return

        print(f"Loading captions from {self.caption_dir}...")
        
        # Debug: show some frame_keys for video datasets
        if self.is_video_dataset and self.file_list:
            print(f"Debug: Sample frame_keys from video dataset '{self.name}':")
            for i, key in enumerate(self.file_list[:5]):
                print(f"  {i}: {key}")
            if len(self.file_list) > 5:
                print(f"  ... and {len(self.file_list) - 5} more")

        json_files = [f for f in os.listdir(self.caption_dir) if f.endswith('.json')]

        read_errors = []
        missing_text_field = []
        invalid_caption_type = []
        non_numeric_ids = []
        unmatched_files = []
        unmatched_filenames = []  # Store actual filenames that couldn't be matched
        empty_caption_files = []
        duplicate_mappings = []

        loaded_count = 0
        total_files = len(json_files)

        for filename in json_files:
            file_path = os.path.join(self.caption_dir, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                read_errors.append((filename, str(e)))
                continue

            if 'text' not in data:
                missing_text_field.append(filename)
                continue

            caption_value = data['text']
            if not isinstance(caption_value, str):
                invalid_caption_type.append((filename, type(caption_value).__name__))
                continue

            if caption_value.strip() == "":
                empty_caption_files.append(filename)
                # Continue loading but record the empty caption

            image_id = os.path.splitext(filename)[0]
            matched_key = None
            if self.is_video_dataset:
                # Filename is a concatenation of <video_id>_<frame_id>
                video_id, sep, frame_base = image_id.rpartition('_')
                if sep == '' or not frame_base:
                    unmatched_files.append((filename, image_id))
                    unmatched_filenames.append(filename)
                    continue
                
                # Debug: show what we're trying to match
                if loaded_count < 3:  # Only show first few for debugging
                    print(f"Debug: Trying to match caption '{filename}' -> video_id='{video_id}', frame_base='{frame_base}'")
                
                # Show what we're looking for in frame_keys
                if loaded_count < 3:
                    print(f"Debug: Looking for frame_key starting with '{video_id}/' and ending with '/{frame_base}.jpg' or '/{frame_base}.png'")
                
                # Show some sample frame_keys that start with this video_id
                matching_video_keys = [key for key in self.file_list if key.startswith(f"{video_id}/")]
                if loaded_count < 3 and matching_video_keys:
                    print(f"Debug: Found {len(matching_video_keys)} frame_keys starting with '{video_id}/':")
                    for i, key in enumerate(matching_video_keys[:3]):
                        print(f"  {i}: {key}")
                    if len(matching_video_keys) > 3:
                        print(f"  ... and {len(matching_video_keys) - 3} more")
                elif loaded_count < 3:
                    print(f"Debug: No frame_keys found starting with '{video_id}/'")
                
                for frame_key in self.file_list:
                    if frame_key == f"{video_id}/{frame_base}.png" or frame_key == f"{video_id}/{frame_base}.jpg":
                        matched_key = frame_key
                        if loaded_count < 3:
                            print(f"Debug: Found match: '{frame_key}'")
                        break
                
            else:
                # Image datasets may use numeric COCO-style IDs
                try:
                    base_name = f"{int(image_id):012d}"
                except Exception:
                    non_numeric_ids.append(filename)
                    continue
                for frame_key in self.file_list:
                    if frame_key == f"{base_name}.jpg" or frame_key == f"{base_name}.png":
                        matched_key = frame_key
                        break

            if matched_key is None:
                unmatched_files.append((filename, image_id))
                unmatched_filenames.append(filename)
                continue

            if matched_key in self.captions:
                duplicate_mappings.append((filename, matched_key))
                # Keep the first mapping and skip overwriting
                continue

            self.captions[matched_key] = caption_value
            loaded_count += 1

        self.has_captions = loaded_count > 0

        # Reporting
        print(f"Loaded {loaded_count} / {total_files} caption files for dataset '{self.name}'.")

        if read_errors:
            print("\nFiles with read/parse errors:")
            for fname, err in read_errors:
                print(f" - {fname}: {err}")

        if missing_text_field:
            print("\nFiles missing 'text' field:")
            for fname in missing_text_field:
                print(f" - {fname}")

        if invalid_caption_type:
            print("\nFiles with invalid caption type (expected string):")
            for fname, t in invalid_caption_type:
                print(f" - {fname}: {t}")

        if non_numeric_ids:
            print("\nFiles with non-numeric IDs (filename without .json not parseable to int):")
            for fname in non_numeric_ids:
                print(f" - {fname}")

        if unmatched_files:
            print("\nCaption files that did not match any dataset frame:")
            for fname, base in unmatched_files:
                print(f" - {fname} (base: {base})")

        if duplicate_mappings:
            print("\nDuplicate caption files mapping to frames that already have captions (kept first, skipped these):")
            for fname, key in duplicate_mappings:
                print(f" - {fname} -> {key}")

        if empty_caption_files:
            print("\nFiles with empty captions (loaded but empty):")
            for fname in empty_caption_files:
                print(f" - {fname}")

        frames_without_captions = len(self.file_list) - len(self.captions)

        # Store summary for later access
        self.caption_load_summary = {
            "total_caption_files": total_files,
            "loaded_captions": loaded_count,
            "read_errors": len(read_errors),
            "missing_text_field": len(missing_text_field),
            "invalid_caption_type": len(invalid_caption_type),
            "non_numeric_ids": len(non_numeric_ids),
            "unmatched_files": len(unmatched_files),
            "unmatched_filenames": unmatched_filenames,  # Store actual filenames
            "duplicate_mappings": len(duplicate_mappings),
            "empty_caption_files": len(empty_caption_files),
            "frames_without_captions": frames_without_captions,
        }

        # Final statistics summary
        print("\n=== Caption Loading Summary ===")
        print(f" Total caption files: {total_files}")
        print(f" Successfully loaded: {loaded_count}")
        print(f" Read/parse errors: {len(read_errors)}")
        print(f" Missing 'text' field: {len(missing_text_field)}")
        print(f" Invalid caption type: {len(invalid_caption_type)}")
        print(f" Non-numeric IDs: {len(non_numeric_ids)}")
        print(f" Unmatched to dataset: {len(unmatched_files)}")
        print(f" Duplicates skipped: {len(duplicate_mappings)}")
        print(f" Empty captions: {len(empty_caption_files)}")
        print(f" Frames without captions: {frames_without_captions}")

    def get_caption(self, frame_key):
        """Get caption for a specific frame"""
        return self.captions.get(frame_key, "")

    def get_all_captions(self):
        """Get all captions"""
        return self.captions.copy()

    def has_caption_for_frame(self, frame_key):
        """Check if a frame has an associated caption"""
        return frame_key in self.captions

    def get_caption_statistics(self):
        """Get statistics about captions including edge-case summary if available"""
        if not self.has_captions and not self.captions:
            return {"has_captions": False}

        caption_lengths = [len(caption) for caption in self.captions.values()]
        empty_captions = sum(1 for caption in self.captions.values() if not caption.strip())

        stats = {
            "has_captions": True,
            "total_captions": len(self.captions),
            "empty_captions": empty_captions,
            "avg_caption_length": sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0,
            "max_caption_length": max(caption_lengths) if caption_lengths else 0,
            "min_caption_length": min(caption_lengths) if caption_lengths else 0,
        }

        if self.caption_load_summary:
            stats.update({f"load_{k}": v for k, v in self.caption_load_summary.items()})

        return stats

    def print_captions(self):
        """Print all captions in the dataset"""
        print(f"\n=== Captions for {self.name} ===")
        for frame_key in sorted(self.captions.keys()):
            caption = self.captions[frame_key]
            print(f"\nFrame: {frame_key}")
            print(f"Caption: {caption}")

    def print_caption_statistics(self):
        """Print caption statistics including edge-case summary if available"""
        stats = self.get_caption_statistics()
        print(f"\n=== Caption Statistics for {self.name} ===")
        if stats.get("has_captions"):
            print(f"Total captions: {stats['total_captions']}")
            print(f"Empty captions: {stats['empty_captions']}")
            print(f"Average caption length: {stats['avg_caption_length']:.2f}")
            print(f"Max caption length: {stats['max_caption_length']}")
            print(f"Min caption length: {stats['min_caption_length']}")

            # Edge-case stats if available
            if self.caption_load_summary:
                print("\n--- Edge-case Loading Stats ---")
                print(f" Total caption files: {self.caption_load_summary.get('total_caption_files', 0)}")
                print(f" Successfully loaded: {self.caption_load_summary.get('loaded_captions', 0)}")
                print(f" Read/parse errors: {self.caption_load_summary.get('read_errors', 0)}")
                print(f" Missing 'text' field: {self.caption_load_summary.get('missing_text_field', 0)}")
                print(f" Invalid caption type: {self.caption_load_summary.get('invalid_caption_type', 0)}")
                print(f" Non-numeric IDs: {self.caption_load_summary.get('non_numeric_ids', 0)}")
                print(f" Unmatched to dataset: {self.caption_load_summary.get('unmatched_files', 0)}")
                print(f" Duplicates skipped: {self.caption_load_summary.get('duplicate_mappings', 0)}")
                print(f" Empty captions: {self.caption_load_summary.get('empty_caption_files', 0)}")
                print(f" Frames without captions: {self.caption_load_summary.get('frames_without_captions', 0)}")
        else:
            print("No captions loaded")

    def parse_caption_mask_ids(self, caption_text):
        """Parse caption text to extract mask IDs referenced in <ids:description> format"""
        import re
        # Find all patterns like <ids:description> where ids are comma-separated integers
        # This handles multiple mentions like <0,1:person> and <2,3:car> in the same caption
        # Also handles whitespace variations like < 1,2 : text > or <1, 2:text>
        pattern = r'<\s*(\d+(?:\s*,\s*\d+)*)\s*:\s*[^>]*>'
        matches = re.findall(pattern, caption_text)
        
        mask_ids = set()
        for match in matches:
            # Split comma-separated IDs and convert to integers
            ids = [int(id_str.strip()) for id_str in match.split(',') if id_str.strip()]
            mask_ids.update(ids)
        
        return sorted(list(mask_ids))

    def check_caption_completeness(self):
        """Check if captions mention all masks in each image"""
        if not self.has_captions:
            return {
                "total_images_with_captions": 0,
                "images_with_incomplete_captions": 0,
                "images_with_complete_captions": 0,
                "incomplete_image_ids": [],
                "completeness_stats": {}
            }
        
        total_images_with_captions = len(self.captions)
        images_with_incomplete_captions = 0
        images_with_complete_captions = 0
        incomplete_image_ids = []
        completeness_stats = {}
        
        for frame_key, caption_text in self.captions.items():
            # Get the number of masks for this image
            segments_info = self.segments_info.get(frame_key, [])
            total_masks = len(segments_info)
            
            if total_masks == 0:
                continue  # Skip images with no masks
            
            # Parse caption to get referenced mask IDs
            referenced_mask_ids = self.parse_caption_mask_ids(caption_text)
            
            # Check if all masks (0 to total_masks-1) are referenced
            expected_mask_ids = set(range(total_masks))
            missing_mask_ids = expected_mask_ids - set(referenced_mask_ids)
            
            is_complete = len(missing_mask_ids) == 0
            
            completeness_stats[frame_key] = {
                "total_masks": total_masks,
                "referenced_mask_ids": referenced_mask_ids,
                "missing_mask_ids": sorted(list(missing_mask_ids)),
                "is_complete": is_complete,
                "caption_text": caption_text[:100] + "..." if len(caption_text) > 100 else caption_text
            }
            
            if is_complete:
                images_with_complete_captions += 1
            else:
                images_with_incomplete_captions += 1
                incomplete_image_ids.append(frame_key)
        
        return {
            "total_images_with_captions": total_images_with_captions,
            "images_with_incomplete_captions": images_with_incomplete_captions,
            "images_with_complete_captions": images_with_complete_captions,
            "incomplete_image_ids": incomplete_image_ids,
            "completeness_stats": completeness_stats
        }

    def print_caption_completeness_analysis(self):
        """Print analysis of caption completeness for all images"""
        completeness_data = self.check_caption_completeness()
        
        print(f"\n=== Caption Completeness Analysis for {self.name} ===")
        print(f"Total images with captions: {completeness_data['total_images_with_captions']}")
        print(f"Images with complete captions: {completeness_data['images_with_complete_captions']}")
        print(f"Images with incomplete captions: {completeness_data['images_with_incomplete_captions']}")
        
        if completeness_data['images_with_incomplete_captions'] > 0:
            print(f"\nImages with incomplete captions (missing some mask IDs):")
            for frame_key in completeness_data['incomplete_image_ids']:
                stats = completeness_data['completeness_stats'][frame_key]
                print(f"  {frame_key}:")
                print(f"    Total masks: {stats['total_masks']}")
                print(f"    Referenced mask IDs: {stats['referenced_mask_ids']}")
                print(f"    Missing mask IDs: {stats['missing_mask_ids']}")
                print(f"    Caption preview: {stats['caption_text']}")
                print()
        
        # Calculate completeness percentage
        if completeness_data['total_images_with_captions'] > 0:
            completeness_pct = (completeness_data['images_with_complete_captions'] / 
                              completeness_data['total_images_with_captions']) * 100
            print(f"Caption completeness rate: {completeness_pct:.1f}%") 