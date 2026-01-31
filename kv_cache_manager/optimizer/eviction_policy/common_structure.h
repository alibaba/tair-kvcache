#pragma once
#include <functional>
#include <memory>
namespace kv_cache_manager {
struct LinkedListNode {
    LinkedListNode *prev = nullptr;
    LinkedListNode *next = nullptr;
    LinkedListNode() = default;
    virtual ~LinkedListNode() = default;
};

class LinkedList {
private:
    LinkedListNode *head_ = nullptr;
    LinkedListNode *tail_ = nullptr;
    size_t size_ = 0;

public:
    LinkedList() {
        head_ = new LinkedListNode();
        tail_ = new LinkedListNode();
        head_->next = tail_;
        tail_->prev = head_;
    }

    ~LinkedList() {
        clear();
        delete head_;
        delete tail_;
    }
    void push_front(LinkedListNode *node) {
        node->next = head_->next;
        node->prev = head_;
        head_->next->prev = node;
        head_->next = node;
        size_++;
    }
    void push_back(LinkedListNode *node) {
        node->prev = tail_->prev;
        node->next = tail_;
        tail_->prev->next = node;
        tail_->prev = node;
        size_++;
    }
    void remove(LinkedListNode *node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        delete node;
        size_--;
    }
    void unlink(LinkedListNode *node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        size_--;
    }
    void move_to_front(LinkedListNode *node) {
        unlink(node);
        push_front(node);
    }
    void move_to_back(LinkedListNode *node) {
        unlink(node);
        push_back(node);
    }
    void insert_sorted(LinkedListNode *node,
                       std::function<bool(const LinkedListNode *, const LinkedListNode *)> compare) {
        LinkedListNode *current = head_->next;
        while (current != tail_ && compare(node, current)) {
            current = current->next;
        }
        node->next = current;
        node->prev = current->prev;
        current->prev->next = node;
        current->prev = node;
        size_++;
    }

    // 从尾部开始遍历的排序插入，适用于插入位置靠近尾部的情况
    void insert_sorted_reverse(LinkedListNode *node,
                               std::function<bool(const LinkedListNode *, const LinkedListNode *)> compare) {
        LinkedListNode *current = tail_->prev;
        while (current != head_ && compare(current, node)) {
            current = current->prev;
        }
        // 插入到 current 之后
        node->prev = current;
        node->next = current->next;
        current->next->prev = node;
        current->next = node;
        size_++;
    }
    LinkedListNode *getHead() const { return head_->next != tail_ ? head_->next : nullptr; }
    LinkedListNode *getTail() const { return tail_->prev != head_ ? tail_->prev : nullptr; }
    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }
    void clear() {
        while (size_ > 0) {
            LinkedListNode *node = head_->next;
            remove(node);
        }
    }
};
} // namespace kv_cache_manager